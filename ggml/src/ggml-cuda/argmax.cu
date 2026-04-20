#include <algorithm>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

// philox-style counter-based PRNG: fast, stateless, deterministic per (seed, counter)
static __device__ __forceinline__ uint32_t philox_hash(uint64_t seed, uint64_t counter) {
    uint32_t lo = (uint32_t)(counter);
    uint32_t hi = (uint32_t)(counter >> 32);
    uint32_t k0 = (uint32_t)(seed);
    uint32_t k1 = (uint32_t)(seed >> 32);
    // 4 rounds of philox-like mixing
    for (int i = 0; i < 4; i++) {
        uint64_t prod = (uint64_t)lo * 0xD2511F53u;
        lo = (uint32_t)(prod >> 32) ^ hi ^ k0;
        hi = (uint32_t)(prod);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return lo;
}

static __device__ __forceinline__ float gumbel_noise(uint64_t seed, int64_t row, int64_t col, int64_t ncols) {
    uint32_t h = philox_hash(seed, row * ncols + col);
    // uniform in (0, 1) — avoid exact 0 and 1
    float u = ((h >> 8) + 0.5f) * (1.0f / 16777216.0f);
    return -logf(-logf(u));
}

// Argmax kernel with optional Gumbel sampling and log-probability output.
// When output_logprob=true, also computes log_prob = log(softmax(logits/temp)[argmax_token])
//   = logits[argmax]/temp - logsumexp(logits/temp)
// using online softmax (single pass over data).
static __global__ void argmax_f32(
        const float * __restrict__ x,
        int32_t * __restrict__ dst,
        const int64_t ncols,
        const int64_t nrows,
        const float inv_temp,
        const uint64_t seed,
        const bool output_logprob) {

    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    // For log-prob: track the max scaled logit and sum of exp(scaled_logit - max)
    float logit_max = -FLT_MAX;  // max of logits[col] * inv_temp (without gumbel)
    float sum_exp = 0.0f;        // running sum for online softmax

    const float * rowx = x + row * ncols;
    const bool use_gumbel = (seed != 0);

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        float scaled = rowx[col] * inv_temp;

        // Online softmax accumulation (always on scaled logits, no gumbel)
        if (output_logprob) {
            if (scaled > logit_max) {
                sum_exp = sum_exp * expf(logit_max - scaled) + 1.0f;
                logit_max = scaled;
            } else {
                sum_exp += expf(scaled - logit_max);
            }
        }

        // Argmax (with optional gumbel perturbation)
        float val = scaled;
        if (use_gumbel) {
            val += gumbel_noise(seed, row, col, ncols);
        }
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    // Warp reduction for argmax
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    // Warp reduction for online softmax (merge logit_max and sum_exp)
    if (output_logprob) {
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            float other_max = __shfl_xor_sync(0xFFFFFFFF, logit_max, offset, WARP_SIZE);
            float other_sum = __shfl_xor_sync(0xFFFFFFFF, sum_exp, offset, WARP_SIZE);
            if (other_max > logit_max) {
                sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                logit_max = other_max;
            } else {
                sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
            }
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        __shared__ float shared_logit_max[max_warps];
        __shared__ float shared_sum_exp[max_warps];

        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
            if (output_logprob) {
                shared_logit_max[warp_id] = logit_max;
                shared_sum_exp[warp_id] = sum_exp;
            }
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
                if (output_logprob) {
                    logit_max = shared_logit_max[lane_id];
                    sum_exp = shared_sum_exp[lane_id];
                }
            } else {
                maxval = -FLT_MAX;
                argmax = -1;
                logit_max = -FLT_MAX;
                sum_exp = 0.0f;
            }

#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }

                if (output_logprob) {
                    float other_max = __shfl_xor_sync(0xFFFFFFFF, logit_max, offset, WARP_SIZE);
                    float other_sum = __shfl_xor_sync(0xFFFFFFFF, sum_exp, offset, WARP_SIZE);
                    if (other_max > logit_max) {
                        sum_exp = sum_exp * expf(logit_max - other_max) + other_sum;
                        logit_max = other_max;
                    } else {
                        sum_exp = sum_exp + other_sum * expf(other_max - logit_max);
                    }
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;

        if (output_logprob) {
            // log_prob = logits[argmax] * inv_temp - (logit_max + log(sum_exp))
            float log_prob = rowx[argmax] * inv_temp - logit_max - logf(sum_exp);
            int32_t prob_bits;
            memcpy(&prob_bits, &log_prob, sizeof(float));
            dst[nrows + row] = prob_bits;
        } else {
            dst[nrows + row] = 0;  // unused but zero-fill for consistency
        }
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    // read temperature and seed from op_params
    float temp = 0.0f;
    uint64_t seed = 0;
    memcpy(&temp, &dst->op_params[0], sizeof(float));
    uint32_t seed_lo = 0, seed_hi = 0;
    memcpy(&seed_lo, &dst->op_params[1], sizeof(uint32_t));
    memcpy(&seed_hi, &dst->op_params[2], sizeof(uint32_t));
    seed = ((uint64_t)seed_hi << 32) | seed_lo;

    const float inv_temp = (temp > 0.0f) ? (1.0f / temp) : 1.0f;
    // always output logprob (dst always has space for 2*nrows elements)
    const bool output_logprob = (temp > 0.0f);

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00, nrows, inv_temp, seed, output_logprob);
}
