#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "ggml-common.h"

// === InnerQ per-channel equalization ===
// Scale K channels before L2 norm + FWHT to reduce quantization error on anisotropic distributions.
// Inverse scale applied to Q in FA kernel to preserve dot products.
// Calibration: accumulate per-channel K^2, then set scale[i] = 1/sqrt(mean(K_i^2) * 128).
static __device__ float d_innerq_channel_scale[128];     // per-channel K scale (init to 1.0)
static __device__ float d_innerq_channel_scale_inv[128]; // per-channel Q inverse scale (init to 1.0)
static __device__ float d_innerq_channel_sq[128];        // calibration accumulator: sum of K_i^2
static __device__ float d_innerq_channel_max[128];       // calibration accumulator: max of |K_i| (for paper's formula)
static __device__ int   d_innerq_count;                  // calibration token count
static __device__ int   d_innerq_calibrate;              // 1 = accumulate stats, 0 = apply scales
static __device__ int   d_innerq_is_k;                   // 1 = current set_rows is K cache, 0 = V cache

// Forward declaration: fattn compilation unit has its own copy of inverse scales
extern void turbo_innerq_update_fattn_scales(const float * scale_inv);
extern void turbo_innerq_init_fattn();

// TCQ error dump: save post-FWHT normalized values and output symbols for autocorrelation analysis
static __device__ float   * d_tcq_dump_x_buf   = nullptr; // [max_groups][128] original values
static __device__ uint8_t * d_tcq_dump_out_buf  = nullptr; // [max_groups][128] output symbols
static __device__ int       d_tcq_dump_max      = 0;       // max groups to dump (0 = disabled)

// === Post-FWHT data extraction for empirical codebook computation ===
// Enabled by TURBO_EXTRACT=<max_samples> env var (e.g. TURBO_EXTRACT=2000000)
// Dumps post-rotation normalized values to /tmp/turbo_postrot.bin (float32)
// Device-visible extraction state
static __device__ float * d_extract_buf_ptr = nullptr;
static __device__ int   * d_extract_pos_ptr = nullptr;
static __device__ int     d_extract_max_val = 0;

// Host-side management
static float * h_extract_gpu_buf = nullptr;
static int   * h_extract_gpu_pos = nullptr;
static int     h_extract_max = 0;
static int     h_extract_state = 0;  // 0=uninit, 1=collecting, 2=done

static void turbo_extract_init(int max_samples) {
	cudaMalloc(&h_extract_gpu_buf, (size_t)max_samples * sizeof(float));
	cudaMalloc(&h_extract_gpu_pos, sizeof(int));
	int zero = 0;
	cudaMemcpy(h_extract_gpu_pos, &zero, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_extract_buf_ptr, &h_extract_gpu_buf, sizeof(float *));
	cudaMemcpyToSymbol(d_extract_pos_ptr, &h_extract_gpu_pos, sizeof(int *));
	cudaMemcpyToSymbol(d_extract_max_val, &max_samples, sizeof(int));
	h_extract_max = max_samples;
	h_extract_state = 1;
	fprintf(stderr, "TURBO_EXTRACT: collecting up to %d post-rotation samples\n", max_samples);
}

static void turbo_extract_check_done() {
	if (h_extract_state != 1) return;
	int pos;
	cudaMemcpy(&pos, h_extract_gpu_pos, sizeof(int), cudaMemcpyDeviceToHost);
	if (pos < h_extract_max) return;
	// Buffer full — dump to disk
	if (pos > h_extract_max) pos = h_extract_max;
	float * host_buf = (float *)malloc((size_t)pos * sizeof(float));
	cudaMemcpy(host_buf, h_extract_gpu_buf, (size_t)pos * sizeof(float), cudaMemcpyDeviceToHost);
	const char * path = "/tmp/turbo_postrot.bin";
	FILE * fp = fopen(path, "wb");
	if (fp) {
		fwrite(host_buf, sizeof(float), pos, fp);
		fclose(fp);
		fprintf(stderr, "TURBO_EXTRACT: wrote %d samples to %s (%.1f MB)\n",
				pos, path, (float)pos * sizeof(float) / (1024*1024));
	}
	free(host_buf);
	// Disable extraction (set device pointers to null)
	float * null_ptr = nullptr;
	int   * null_iptr = nullptr;
	int     zero_max = 0;
	cudaMemcpyToSymbol(d_extract_buf_ptr, &null_ptr, sizeof(float *));
	cudaMemcpyToSymbol(d_extract_pos_ptr, &null_iptr, sizeof(int *));
	cudaMemcpyToSymbol(d_extract_max_val, &zero_max, sizeof(int));
	cudaFree(h_extract_gpu_buf); h_extract_gpu_buf = nullptr;
	cudaFree(h_extract_gpu_pos); h_extract_gpu_pos = nullptr;
	h_extract_state = 2;
}

// Device-side: append 128 post-rotation values to extraction buffer
static __device__ void turbo_extract_append(const float * x) {
	if (!d_extract_buf_ptr || !d_extract_pos_ptr) return;
	int base = atomicAdd(d_extract_pos_ptr, 128);
	if (base + 128 <= d_extract_max_val) {
		for (int j = 0; j < 128; j++) d_extract_buf_ptr[base + j] = x[j];
	}
}

// Host-side init: set identity scales, zero accumulators
static void turbo_innerq_init() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    float zeros[128] = {};
    int zero = 0;
    cudaMemcpyToSymbol(d_innerq_channel_scale, ones, sizeof(ones));
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv, ones, sizeof(ones));
    cudaMemcpyToSymbol(d_innerq_channel_sq, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_channel_max, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_innerq_calibrate, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_innerq_is_k, &zero, sizeof(zero));
    turbo_innerq_init_fattn();
}

// Host-side: set K/V flag before kernel launch (called from set-rows.cu)
static void turbo_innerq_set_is_k(int is_k) {
    cudaMemcpyToSymbol(d_innerq_is_k, &is_k, sizeof(int));
}

// Host-side: enable calibration mode
static void turbo_innerq_start_calibration() {
    float zeros[128] = {};
    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_channel_sq, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_channel_max, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(zero));
    cudaMemcpyToSymbol(d_innerq_calibrate, &one, sizeof(one));
}

// Host-side: finalize calibration — compute scales from accumulated stats
static void turbo_innerq_finalize_calibration() {
    int zero = 0;
    cudaMemcpyToSymbol(d_innerq_calibrate, &zero, sizeof(zero));

    float sq[128], ch_max[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_innerq_channel_sq, sizeof(sq));
    cudaMemcpyFromSymbol(ch_max, d_innerq_channel_max, sizeof(ch_max));
    cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(count));

    if (count == 0) return;

    // Mode: 0=RMS-based (default), 1=max-based (paper's formula: sqrt(max|K_i|))
    static const char * mode_env = getenv("TURBO_INNERQ_MODE");
    int mode = mode_env ? atoi(mode_env) : 0;

    static const char * strength_env = getenv("TURBO_INNERQ_STRENGTH");
    float strength = strength_env ? atof(strength_env) : 0.5f;
    float max_clamp = 2.0f;

    float scale[128], scale_inv[128];
    float max_ratio = 1.0f;

    if (mode == 1) {
        // Paper's formula: scale[i] = 1/sqrt(max(|K_{:,i}|))
        // This normalizes each channel so its max value becomes sqrt(max_val)
        fprintf(stderr, "InnerQ mode=1 (paper's max-based formula)\n");
        for (int i = 0; i < 128; i++) {
            if (ch_max[i] > 1e-10f) {
                float s = 1.0f / sqrtf(ch_max[i]);
                // Normalize so mean scale = 1 (preserve overall magnitude)
                scale[i] = s;
            } else {
                scale[i] = 1.0f;
            }
        }
        // Normalize scales to have geometric mean ≈ 1
        float log_sum = 0.0f;
        for (int i = 0; i < 128; i++) log_sum += logf(scale[i]);
        float geo_mean = expf(log_sum / 128.0f);
        for (int i = 0; i < 128; i++) {
            scale[i] /= geo_mean;
            if (scale[i] > max_clamp) scale[i] = max_clamp;
            if (scale[i] < 1.0f / max_clamp) scale[i] = 1.0f / max_clamp;
            scale_inv[i] = 1.0f / scale[i];
            float ratio = fmaxf(scale[i], 1.0f / scale[i]);
            if (ratio > max_ratio) max_ratio = ratio;
        }
    } else {
        // RMS-based: scale = (mean_rms/channel_rms)^strength
        float total_rms = 0.0f;
        float channel_rms[128];
        for (int i = 0; i < 128; i++) {
            channel_rms[i] = sqrtf(sq[i] / count);
            total_rms += channel_rms[i];
        }
        float mean_rms = total_rms / 128.0f;

        for (int i = 0; i < 128; i++) {
            if (channel_rms[i] > 1e-10f) {
                float raw = mean_rms / channel_rms[i];
                float s = powf(raw, strength);
                if (s > max_clamp) s = max_clamp;
                if (s < 1.0f / max_clamp) s = 1.0f / max_clamp;
                scale[i] = s;
                scale_inv[i] = 1.0f / s;
            } else {
                scale[i] = 1.0f;
                scale_inv[i] = 1.0f;
            }
            float ratio = fmaxf(scale[i], 1.0f / scale[i]);
            if (ratio > max_ratio) max_ratio = ratio;
        }
    }

    fprintf(stderr, "InnerQ calibration: %d tokens, mode=%d, strength=%.2f, max scale ratio: %.3f (clamped to %.1f)\n",
            count, mode, strength, max_ratio, max_clamp);

    // Auto-detect: if channels are already well-balanced, InnerQ won't help — skip
    if (max_ratio < 1.2f) {
        fprintf(stderr, "InnerQ: max ratio %.3f < 1.2 — channels already balanced, disabling (would hurt quality)\n", max_ratio);
        float ones[128];
        for (int i = 0; i < 128; i++) ones[i] = 1.0f;
        cudaMemcpyToSymbol(d_innerq_channel_scale, ones, sizeof(ones));
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv, ones, sizeof(ones));
        turbo_innerq_update_fattn_scales(ones);
        return;
    }

    // Print top-5 most affected channels
    float scale_copy[128];
    for (int i = 0; i < 128; i++) scale_copy[i] = scale[i];
    for (int k = 0; k < 5; k++) {
        float best = 0; int best_i = -1;
        for (int i = 0; i < 128; i++) {
            float r = fabsf(logf(scale_copy[i]));
            if (r > best) { best = r; best_i = i; }
        }
        if (best_i >= 0) {
            fprintf(stderr, "  channel %d: scale=%.4f (max=%.6f, rms=%.6f)\n",
                    best_i, scale[best_i], ch_max[best_i], sqrtf(sq[best_i] / count));
            scale_copy[best_i] = 1.0f; // mark as printed
        }
    }

    cudaMemcpyToSymbol(d_innerq_channel_scale, scale, sizeof(scale));
    cudaMemcpyToSymbol(d_innerq_channel_scale_inv, scale_inv, sizeof(scale_inv));
    turbo_innerq_update_fattn_scales(scale_inv);
}

// === Shared constants ===
static __constant__ float d_turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float d_turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// === TURBO2: 2-bit codebook (Lloyd-Max for N(0, 1/128)) ===
static __constant__ float d_turbo_centroids_2bit[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};
static __constant__ float d_turbo_mid_2bit[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// === TURBO4: 4-bit codebook (Lloyd-Max for N(0, 1/sqrt(128))) ===
static __constant__ float d_turbo_centroids_4bit[16] = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};
static __constant__ float d_turbo_mid_4bit[15] = {
    -0.212232f, -0.162977f, -0.127056f, -0.097191f, -0.070693f,
    -0.046190f, -0.022832f,  0.000000f,  0.022832f,  0.046190f,
     0.070693f,  0.097191f,  0.127056f,  0.162977f,  0.212232f,
};

// === FWHT rotation sign arrays (from turbo-wht.h, seed=42 rotation, seed=1042 QJL) ===
static __constant__ float d_turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
static __constant__ float d_turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};
// QJL sign arrays removed — turbo4 now uses pure 4-bit PolarQuant (no QJL correction)

// === FWHT rotation functions ===
static __device__ __forceinline__
void turbo_fwht_128_cuda(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) x[i] *= inv_sqrt_128;
}

// Forward rotation: signs1 → FWHT → signs2
static __device__ __forceinline__
void turbo_rotate_forward_cuda(float * x, const float * s1, const float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128_cuda(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

static __device__ __forceinline__
uint8_t turbo_find_nearest_3bit(float val) {
    if      (val < d_turbo_mid_3bit[0]) return 0;
    else if (val < d_turbo_mid_3bit[1]) return 1;
    else if (val < d_turbo_mid_3bit[2]) return 2;
    else if (val < d_turbo_mid_3bit[3]) return 3;
    else if (val < d_turbo_mid_3bit[4]) return 4;
    else if (val < d_turbo_mid_3bit[5]) return 5;
    else if (val < d_turbo_mid_3bit[6]) return 6;
    else                                return 7;
}

static __device__ __forceinline__
uint8_t turbo_find_nearest_4bit(float val) {
    // Binary search over 15 midpoints for 16 centroids
    if (val < d_turbo_mid_4bit[7]) {
        if (val < d_turbo_mid_4bit[3]) {
            if (val < d_turbo_mid_4bit[1]) {
                return val < d_turbo_mid_4bit[0] ? 0 : 1;
            } else {
                return val < d_turbo_mid_4bit[2] ? 2 : 3;
            }
        } else {
            if (val < d_turbo_mid_4bit[5]) {
                return val < d_turbo_mid_4bit[4] ? 4 : 5;
            } else {
                return val < d_turbo_mid_4bit[6] ? 6 : 7;
            }
        }
    } else {
        if (val < d_turbo_mid_4bit[11]) {
            if (val < d_turbo_mid_4bit[9]) {
                return val < d_turbo_mid_4bit[8] ? 8 : 9;
            } else {
                return val < d_turbo_mid_4bit[10] ? 10 : 11;
            }
        } else {
            if (val < d_turbo_mid_4bit[13]) {
                return val < d_turbo_mid_4bit[12] ? 12 : 13;
            } else {
                return val < d_turbo_mid_4bit[14] ? 14 : 15;
            }
        }
    }
}

// === TURBO3: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo3(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int innerq_is_k,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO3_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_0 * dst_row_ptr = (block_turbo3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO3_GROUP;
    const int blocks_per_group = QK_TURBO3_GROUP / QK_TURBO3;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    // InnerQ: calibrate from both K and V, apply scaling to both
    if (d_innerq_calibrate) {
        for (int j = 0; j < 128; j++) {
            atomicAdd(&d_innerq_channel_sq[j], x[j] * x[j]);
            float abs_val = fabsf(x[j]);
            // atomicMax for float: CAS loop (no native float atomicMax)
            unsigned int * addr = (unsigned int *)&d_innerq_channel_max[j];
            unsigned int old_val = __float_as_uint(abs_val);
            unsigned int assumed;
            do {
                assumed = *addr;
                if (__uint_as_float(assumed) >= abs_val) break;
            } while (atomicCAS(addr, assumed, old_val) != assumed);
        }
        atomicAdd(&d_innerq_count, 1);
    }
    for (int j = 0; j < 128; j++) x[j] *= d_innerq_channel_scale[j];
    norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += x[j] * x[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Post-rotation extraction (if enabled)
    turbo_extract_append(x);
    // Quantize and accumulate reconstruction norm for correction
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo3_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO3;
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk.signs[j] = 0;
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t idx = turbo_find_nearest_3bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));
            float c = d_turbo_centroids_3bit[idx];
            recon_norm_sq += c * c;
        }
    }
    // Norm correction: store corrected norm so dequant(x) has exact original L2 norm
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_row_ptr[grp_idx * blocks_per_group + b].norm = __float2half(corrected_norm);
    }
}

// === TURBO3: GET_ROWS dequantize ===
#define QR_TURBO3_0 2
static __device__ __forceinline__
void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_0 * x = (const block_turbo3_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.x = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
    { const int j = iqs + 16;
      const uint8_t low2 = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      const uint8_t hi1  = (x[ib].signs[j/8] >> (j%8)) & 0x1;
      v.y = d_turbo_centroids_3bit[low2 | (hi1 << 2)] * norm; }
}

// Temperature scaling for turbo4. Override via TURBO4_ALPHA env var.
static __constant__ float d_turbo4_alpha = 1.2f;

// === TURBO4: SET_ROWS quantize (4-bit PolarQuant, no QJL) ===
static __device__ __forceinline__
void quantize_f32_turbo4_0_block(const float * src, block_turbo4_0 * dst) {
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    // Forward FWHT rotation before quantization
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    // Post-rotation extraction (if enabled)
    turbo_extract_append(x);
    // 4-bit quantization: find nearest of 16 centroids, pack 2 per byte
    for (int j = 0; j < 128; j += 2) {
        uint8_t idx0 = turbo_find_nearest_4bit(x[j]);
        uint8_t idx1 = turbo_find_nearest_4bit(x[j + 1]);
        dst->qs[j / 2] = (idx1 << 4) | idx0;
    }
    // Norm correction: compute reconstruction norm in rotated space
    float recon_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (j & 1) ? (dst->qs[j / 2] >> 4) : (dst->qs[j / 2] & 0xF);
        float r = d_turbo_centroids_4bit[idx];
        recon_sq += r * r;
    }
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
    dst->norm = __float2half(corrected * d_turbo4_alpha);
}

// === TURBO4: GET_ROWS dequantize ===
#define QR_TURBO4_0 2
static __device__ __forceinline__
void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo4_0 * x = (const block_turbo4_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.x = d_turbo_centroids_4bit[idx] * norm; }
    { const int j = iqs + 64;
      uint8_t idx = (j & 1) ? (x[ib].qs[j / 2] >> 4) : (x[ib].qs[j / 2] & 0xF);
      v.y = d_turbo_centroids_4bit[idx] * norm; }
}

// === TURBO2: find nearest 2-bit centroid ===
static __device__ __forceinline__
uint8_t turbo_find_nearest_2bit(float val) {
    if      (val < d_turbo_mid_2bit[0]) return 0;
    else if (val < d_turbo_mid_2bit[1]) return 1;
    else if (val < d_turbo_mid_2bit[2]) return 2;
    else                                return 3;
}

// === TURBO2: SET_ROWS kernel ===
template<typename idx_t>
static __global__ void k_set_rows_turbo2(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_0 * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i >= ne_total_groups) return;
    const int64_t i_base = i * QK_TURBO2_GROUP;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_0 * dst_row_ptr = (block_turbo2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    const int grp_idx = i00 / QK_TURBO2_GROUP;
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2;
    float x[128]; float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) { x[j] = grp_src[j]; norm_sq += x[j] * x[j]; }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;
    for (int j = 0; j < 128; j++) x[j] *= inv_norm;
    turbo_rotate_forward_cuda(x, d_turbo_wht_signs1, d_turbo_wht_signs2);
    float recon_norm_sq = 0.0f;
    for (int b = 0; b < blocks_per_group; b++) {
        block_turbo2_0 & blk = dst_row_ptr[grp_idx * blocks_per_group + b];
        const int off = b * QK_TURBO2;
        for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = turbo_find_nearest_2bit(x[off + j]);
            blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            float c = d_turbo_centroids_2bit[idx];
            recon_norm_sq += c * c;
        }
    }
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    for (int b = 0; b < blocks_per_group; b++) {
        dst_row_ptr[grp_idx * blocks_per_group + b].norm = __float2half(corrected_norm);
    }
}

// === TURBO2: GET_ROWS dequantize ===
#define QR_TURBO2_0 2
static __device__ __forceinline__
void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_0 * x = (const block_turbo2_0 *)vx;
    const float norm = __half2float(x[ib].norm);
    { const int j = iqs;
      const uint8_t idx = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      v.x = d_turbo_centroids_2bit[idx] * norm; }
    { const int j = iqs + 16;
      const uint8_t idx = (x[ib].qs[j/4] >> ((j%4)*2)) & 0x3;
      v.y = d_turbo_centroids_2bit[idx] * norm; }
}

// === TURBO3_TCQ: Trellis-Coded Quantization (right-shift bitshift trellis, k=3, L=9) ===
// MSE reduction: 50.1% vs Lloyd-Max 3-bit, +3.02 dB. numpy GLA: n_train=4000, 100 iters, seed=99. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turbo3_tcq_codebook[512] = {
    -0.24244059f, -0.12586778f, -0.06693592f, -0.02260770f, +0.01492950f, +0.05467265f, +0.10069778f, +0.18883320f,
    -0.19693744f, -0.14152811f, -0.09539399f, -0.06046141f, -0.02731707f, +0.01163860f, +0.05423523f, +0.11278591f,
    -0.11856443f, -0.06727399f, -0.02913110f, +0.00417571f, +0.03549468f, +0.07371171f, +0.11926779f, +0.18401266f,
    -0.25362726f, -0.15759121f, -0.10456934f, -0.06284792f, -0.01789622f, +0.03435958f, +0.08292559f, +0.14658904f,
    -0.16766223f, -0.09932603f, -0.04795861f, -0.00316137f, +0.03350896f, +0.07203513f, +0.12375449f, +0.24558071f,
    -0.21340639f, -0.11273975f, -0.05969454f, -0.02112451f, +0.01584557f, +0.05606037f, +0.09979239f, +0.18008010f,
    -0.14495838f, -0.08746232f, -0.05134764f, -0.02051995f, +0.00527687f, +0.03116450f, +0.06451037f, +0.11952747f,
    -0.20422562f, -0.11092815f, -0.05362599f, -0.00892124f, +0.02997769f, +0.07779223f, +0.13904265f, +0.22305706f,
    -0.17846867f, -0.11931835f, -0.08258449f, -0.04965282f, -0.01742237f, +0.01840734f, +0.06431029f, +0.13633489f,
    -0.16413829f, -0.09193468f, -0.05326458f, -0.01892892f, +0.02155236f, +0.07144441f, +0.12140869f, +0.16414391f,
    -0.16097247f, -0.11795594f, -0.07439681f, -0.03746189f, -0.00202306f, +0.03708065f, +0.07964572f, +0.25785580f,
    -0.21079398f, -0.08501953f, -0.04986830f, -0.02919976f, -0.00307999f, +0.01172143f, +0.04960671f, +0.10403022f,
    -0.13488377f, -0.08804465f, -0.05803939f, -0.02886309f, -0.00121364f, +0.03075502f, +0.07380433f, +0.14234027f,
    -0.17733476f, -0.11930768f, -0.08073044f, -0.05102654f, -0.02174008f, +0.01207697f, +0.05188434f, +0.10528153f,
    -0.27424359f, -0.14814219f, -0.09042648f, -0.04750653f, -0.00688004f, +0.03821837f, +0.08375420f, +0.15444848f,
    -0.17072668f, -0.11573062f, -0.07891619f, -0.04997802f, -0.02360069f, +0.00884181f, +0.04775132f, +0.09702020f,
    -0.13396922f, -0.08187833f, -0.03989934f, -0.00285008f, +0.03193579f, +0.06714261f, +0.10630646f, +0.19974332f,
    -0.10977794f, -0.05588607f, -0.01988312f, +0.00588292f, +0.02463065f, +0.04931722f, +0.08140395f, +0.11857282f,
    -0.11285258f, -0.06842930f, -0.03478571f, +0.00135103f, +0.04282236f, +0.08846240f, +0.14403294f, +0.18865710f,
    -0.16570802f, -0.13114756f, -0.08916780f, -0.01495983f, +0.02156897f, +0.05788230f, +0.10420620f, +0.15807896f,
    -0.09603385f, -0.05330852f, -0.01872682f, +0.01128407f, +0.04181543f, +0.07901397f, +0.12809893f, +0.20628030f,
    -0.12671234f, -0.07713382f, -0.04176560f, -0.01075576f, +0.02093381f, +0.05861618f, +0.10125964f, +0.16253341f,
    -0.12186792f, -0.07046833f, -0.02827389f, +0.00582941f, +0.03785510f, +0.07531738f, +0.13185618f, +0.20784822f,
    -0.11580890f, -0.06750206f, -0.03211596f, -0.00041264f, +0.02880913f, +0.06547855f, +0.11221221f, +0.17096693f,
    -0.20808545f, -0.15288957f, -0.09920800f, -0.05654906f, -0.02077297f, +0.01662349f, +0.06161885f, +0.11496038f,
    -0.25925224f, -0.12740968f, -0.07758909f, -0.03847224f, -0.00659505f, +0.02506258f, +0.05676728f, +0.15852313f,
    -0.20711072f, -0.15256361f, -0.09078260f, -0.04651003f, -0.01428200f, +0.02046691f, +0.06122406f, +0.11168941f,
    -0.29227489f, -0.10113064f, -0.06318919f, -0.04224788f, -0.01237292f, +0.01916771f, +0.05288843f, +0.08860565f,
    -0.18939137f, -0.13610712f, -0.07454449f, -0.03508454f, -0.00070383f, +0.03791586f, +0.07589655f, +0.12249952f,
    -0.23639095f, -0.16088664f, -0.10112434f, -0.05671202f, -0.02441828f, +0.01108337f, +0.04704943f, +0.08991196f,
    -0.18832129f, -0.10863860f, -0.06105676f, -0.02453765f, +0.00571738f, +0.03372482f, +0.06261604f, +0.10699216f,
    -0.22723058f, -0.15697967f, -0.09283338f, -0.04977757f, -0.00658221f, +0.03587560f, +0.07948679f, +0.13286804f,
    -0.10523734f, -0.05894853f, -0.01794997f, +0.01681460f, +0.05235468f, +0.08761997f, +0.12857652f, +0.27355651f,
    -0.16195214f, -0.08037403f, -0.03931148f, -0.00205822f, +0.03885316f, +0.08372425f, +0.14362726f, +0.19498548f,
    -0.11559617f, -0.06571012f, -0.02964597f, -0.00045770f, +0.02920656f, +0.06525015f, +0.11007642f, +0.23265810f,
    -0.12326290f, -0.06516271f, -0.02775460f, +0.00911453f, +0.03682196f, +0.07574877f, +0.13758171f, +0.19163566f,
    -0.09889923f, -0.05620999f, -0.01514455f, +0.01793674f, +0.05562053f, +0.10430135f, +0.16772700f, +0.28700828f,
    -0.14117142f, -0.08234416f, -0.03966629f, -0.00272311f, +0.03102731f, +0.07227346f, +0.13315912f, +0.20565525f,
    -0.09793990f, -0.05264642f, -0.01436317f, +0.01968854f, +0.05324087f, +0.09480734f, +0.16667446f, +0.25740325f,
    -0.14365898f, -0.07946859f, -0.03025317f, +0.01447767f, +0.05407316f, +0.09543498f, +0.14146231f, +0.20799392f,
    -0.16657843f, -0.10643959f, -0.06051657f, -0.02209583f, +0.01260932f, +0.04745538f, +0.09038523f, +0.16133716f,
    -0.21383845f, -0.13881313f, -0.09221762f, -0.05544837f, -0.02178388f, +0.01677356f, +0.05674765f, +0.10728363f,
    -0.17472305f, -0.11292139f, -0.06834519f, -0.03219563f, +0.00094835f, +0.03451309f, +0.07811368f, +0.14950613f,
    -0.21735978f, -0.14172379f, -0.09016410f, -0.05325706f, -0.02099085f, +0.01431495f, +0.05746740f, +0.10986551f,
    -0.16108559f, -0.09852512f, -0.05524211f, -0.01762269f, +0.01394665f, +0.05029779f, +0.09104291f, +0.15619289f,
    -0.18963714f, -0.12396694f, -0.07575205f, -0.03500398f, -0.00238001f, +0.03088680f, +0.06744511f, +0.11232874f,
    -0.15580266f, -0.11168178f, -0.07526547f, -0.04145918f, -0.00974866f, +0.03212880f, +0.07638067f, +0.13532050f,
    -0.18869418f, -0.12704822f, -0.07090112f, -0.03539131f, -0.00940597f, +0.01779585f, +0.05332254f, +0.10070462f,
    -0.10802900f, -0.05559649f, -0.01134203f, +0.02766773f, +0.06135347f, +0.09766156f, +0.13701990f, +0.20283839f,
    -0.15064489f, -0.08763143f, -0.05088234f, -0.01813378f, +0.01489159f, +0.05492927f, +0.10086069f, +0.16056357f,
    -0.10806098f, -0.05308804f, -0.01607634f, +0.01716060f, +0.04692414f, +0.08323829f, +0.12591397f, +0.19397456f,
    -0.14325471f, -0.07795846f, -0.03858727f, -0.01405432f, +0.01490734f, +0.04949452f, +0.09137284f, +0.14710410f,
    -0.08884655f, -0.04006506f, +0.00188640f, +0.03342239f, +0.06599921f, +0.10063411f, +0.13860981f, +0.21032288f,
    -0.12697365f, -0.06556813f, -0.02598349f, +0.00936226f, +0.04159310f, +0.07441500f, +0.11134150f, +0.15887714f,
    -0.24867819f, -0.09192626f, -0.04786854f, -0.01128089f, +0.02160387f, +0.05966909f, +0.10629620f, +0.19008696f,
    -0.10229637f, -0.05171858f, -0.01251010f, +0.01547078f, +0.03557770f, +0.06102134f, +0.09990518f, +0.15643583f,
    -0.22377374f, -0.14769778f, -0.08583978f, -0.04202831f, -0.00770624f, +0.02992121f, +0.07128810f, +0.12562713f,
    -0.11691130f, -0.05988247f, -0.02030350f, +0.01373520f, +0.04911441f, +0.09225391f, +0.15575270f, +0.23753035f,
    -0.18158022f, -0.09763482f, -0.05659763f, -0.02414636f, +0.00537057f, +0.03960274f, +0.07579990f, +0.12346489f,
    -0.26066830f, -0.12511689f, -0.06579913f, -0.00571045f, +0.03891529f, +0.08188405f, +0.12671439f, +0.18494135f,
    -0.15464623f, -0.08975428f, -0.04408587f, -0.01101469f, +0.02199709f, +0.05924839f, +0.10465596f, +0.26287255f,
    -0.19226125f, -0.11309006f, -0.07365324f, -0.03543059f, -0.00178878f, +0.03501295f, +0.07791925f, +0.14937145f,
    -0.12649273f, -0.06018269f, -0.01573098f, +0.02200219f, +0.05903495f, +0.09840808f, +0.13520581f, +0.18245036f,
    -0.16474872f, -0.09278035f, -0.04699890f, -0.00779894f, +0.03187623f, +0.07828258f, +0.13561429f, +0.23917313f
};

// Asymmetric temperature scaling for TCQ norm. V-heavier scaling improves long-context quality.
// Override via TURBO_TCQ_ALPHA (K) and TURBO_TCQ_ALPHA_V (V) env vars.
// If TURBO_TCQ_ALPHA_V is not set, V uses the same alpha as K.
static __constant__ float d_tcq_norm_alpha = 1.1f;
static __constant__ float d_tcq_norm_alpha_v = 1.3f;

// TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis
// 512 threads per block (one per trellis state), one block per 128-element group
// Backtrace stored in shared memory (32KB, 4-bit packed)
template<typename idx_t>
static __global__ void __launch_bounds__(512, 1) k_set_rows_turbo3_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo3_tcq * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int innerq_is_k,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int64_t group = blockIdx.x;
    if (group >= ne_total_groups) return;

    const int sid = threadIdx.x; // state index 0..511

    // Compute source and destination pointers (same index math as turbo3)
    const int64_t i_base = group * QK_TURBO3_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo3_tcq * dst_blk = (block_turbo3_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                                  + (i00 / QK_TURBO3_TCQ);

    // Shared memory layout:
    // x[128]     : rotated+normalized input (also reused as outputs[] after Viterbi)
    // cost[512]  : current path costs (also reused for reductions)
    // bt[128][256]: backtrace, 4-bit packed (best predecessor index 0-7)
    __shared__ float x[128];
    __shared__ float cost[512];
    __shared__ uint8_t bt[128][256]; // 32KB: bt[t][s/2] = (pred_s_even) | (pred_s_odd << 4)
    __shared__ int warp_min_idx[16];
    __shared__ float warp_min_cost[16];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    if (d_innerq_calibrate && sid < 128) {
        atomicAdd(&d_innerq_channel_sq[sid], x[sid] * x[sid]);
        float abs_val = fabsf(x[sid]);
        unsigned int * addr = (unsigned int *)&d_innerq_channel_max[sid];
        unsigned int old_val = __float_as_uint(abs_val);
        unsigned int assumed;
        do {
            assumed = *addr;
            if (__uint_as_float(assumed) >= abs_val) break;
        } while (atomicCAS(addr, assumed, old_val) != assumed);
        if (sid == 0) atomicAdd(&d_innerq_count, 1);
    }
    // No sync needed: calibration writes d_innerq_channel_sq/max, scaling reads d_innerq_channel_scale
    if (sid < 128) x[sid] *= d_innerq_channel_scale[sid];
    __syncthreads();

    // Norm reduction
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 256; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT
    if (sid < 128) x[sid] *= d_turbo_wht_signs1[sid];
    __syncthreads();
    for (int h = 1; h < 128; h *= 2) {
        if (sid < 64) {
            int j = (sid / h) * (2 * h) + (sid % h);
            float a = x[j], b = x[j + h];
            x[j] = a + b; x[j + h] = a - b;
        }
        __syncthreads();
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * d_turbo_wht_signs2[sid];
    __syncthreads();

    if (sid == 0) turbo_extract_append(x);
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Viterbi forward pass: free initial state (all equally viable)
    cost[sid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        float xt = x[t];

        // Right-shift trellis: ns = (prev >> 3) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 3) | p, for p = 0..7
        int base_prev = (sid & 0x3F) << 3;
        float dist = xt - d_turbo3_tcq_codebook[sid];
        dist = dist * dist;

        float best = 1e30f;
        int best_p = 0;
        for (int p = 0; p < 8; p++) {
            float c = cost[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }

        __syncthreads();
        cost[sid] = best + dist;

        // Store backtrace: 4-bit packed, 2 entries per byte
        if (sid % 2 == 0) {
            bt[t][sid / 2] = (uint8_t)best_p;
        }
        __syncthreads();
        if (sid % 2 == 1) {
            bt[t][sid / 2] |= ((uint8_t)best_p) << 4;
        }
        __syncthreads();
    }

    // Warp argmin over 512 costs
    {
        float my_cost = cost[sid];
        int my_idx = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid == 0) {
        float best = warp_min_cost[0];
        int best_idx = warp_min_idx[0];
        for (int w = 1; w < 16; w++) {
            if (warp_min_cost[w] < best) { best = warp_min_cost[w]; best_idx = warp_min_idx[w]; }
        }
        shared_initial_state = best_idx; // temporarily: best final state (becomes initial after backtrack)
    }
    __syncthreads();

    // Save x[] to global buffer before backtrack overwrites it
    if (d_tcq_dump_max > 0 && group < d_tcq_dump_max && sid < 128)
        d_tcq_dump_x_buf[group * 128 + sid] = x[sid];

    // Backtrack (inherently sequential)
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6);
            int p = (bt[t][state / 2] >> ((state % 2) * 4)) & 0xF;
            state = ((state & 0x3F) << 3) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Save output symbols to global buffer
    if (d_tcq_dump_max > 0 && group < d_tcq_dump_max && sid < 128)
        d_tcq_dump_out_buf[group * 128 + sid] = outputs[sid];

    // Parallel recon norm: t>=2 can compute state directly from 3 outputs (3 shifts of 3 = 9 bits)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 2) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 3) | (((int)outputs[t]) << 6);
        } else {
            cur_state = ((int)outputs[sid - 2] & 0x7)
                      | (((int)outputs[sid - 1] & 0x7) << 3)
                      | (((int)outputs[sid]     & 0x7) << 6);
        }
        float c = d_turbo3_tcq_codebook[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 256; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= innerq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Serial bitpack — byte-alignment prevents parallel atomicOr
    if (sid == 0) {
        for (int j = 0; j < 49; j++) dst_blk->qs[j] = 0;
        dst_blk->qs[0] = (uint8_t)((shared_initial_state >> 3) & 0x3F);
        for (int t = 0; t < 128; t++) {
            const int bit_pos = 6 + t * 3;
            const int byte_idx = bit_pos / 8;
            const int bit_off = bit_pos % 8;
            const int out = outputs[t] & 0x7;
            dst_blk->qs[byte_idx] |= (uint8_t)(out << bit_off);
            if (bit_off > 5) dst_blk->qs[byte_idx + 1] |= (uint8_t)(out >> (8 - bit_off));
        }
        dst_blk->norm = __float2half(corrected_norm);
    }
}

// TCQ GET_ROWS dequantize (for non-FA paths)
#define QR_TURBO3_TCQ 2
static __device__ __forceinline__
void dequantize_turbo3_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs
    {
        const int t = iqs;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.x = d_turbo3_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64 (stride = half block size)
    {
        const int t = iqs + 64;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.y = d_turbo3_tcq_codebook[state] * norm;
    }
}

// =====================================================================================
// TURBO2_TCQ: 2-bit Trellis-Coded Quantization (k=2, L=8, 256 states, free initial state)
// =====================================================================================

// MSE reduction: 33.1% vs Lloyd-Max 2-bit, +1.75 dB. numpy GLA: n_train=4000, 100 iters, 5 restarts. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turbo2_tcq_codebook[256] = {
    -0.08176727f, -0.00033508f, +0.06850938f, +0.16613583f, -0.14090237f, -0.05715980f, +0.01615283f, +0.11012612f,
    -0.10581727f, -0.04260033f, -0.00423828f, +0.06296677f, -0.17352516f, -0.07213694f, +0.02485547f, +0.10813029f,
    -0.12736021f, -0.06026637f, +0.00177779f, +0.06987048f, -0.08498892f, -0.01943354f, +0.06211906f, +0.01397950f,
    -0.17903381f, -0.01989968f, +0.03569642f, +0.09051796f, -0.09042171f, -0.02577177f, +0.02050355f, +0.10467158f,
    -0.21265116f, -0.11087410f, -0.04349163f, +0.01669601f, -0.12012258f, -0.01521601f, +0.07030928f, +0.13750617f,
    -0.06601736f, -0.04198077f, +0.02279012f, +0.10377382f, -0.07896508f, -0.00657534f, +0.06652649f, +0.17177304f,
    -0.07452555f, -0.00981928f, +0.04254026f, +0.11680857f, -0.12769225f, -0.04400226f, +0.01111500f, +0.08063783f,
    -0.05339707f, +0.01173677f, +0.07039803f, +0.14338760f, -0.12492259f, -0.05478338f, -0.01731757f, +0.04320757f,
    -0.00530445f, -0.15542837f, -0.06801344f, +0.04485723f, -0.07050634f, +0.01234248f, +0.11757696f, +0.22165567f,
    -0.01849510f, +0.04277446f, +0.08655161f, +0.15533215f, -0.10084474f, -0.00810490f, -0.03715962f, +0.04786975f,
    -0.02117090f, +0.04766359f, +0.08838871f, +0.16277327f, -0.24295192f, -0.12420259f, -0.05557786f, +0.12114887f,
    -0.12861997f, -0.06805481f, -0.05590313f, +0.01283404f, -0.01349204f, +0.05466014f, +0.10226475f, +0.19152307f,
    -0.09299547f, -0.02196216f, +0.03284279f, +0.09021873f, -0.07505369f, +0.08066312f, -0.03999974f, +0.04350512f,
    +0.00485651f, +0.05240202f, +0.12679257f, +0.19781399f, -0.18016882f, -0.11454904f, -0.06387294f, +0.01354196f,
    -0.17339253f, -0.10154387f, -0.03942726f, +0.03053090f, -0.01029367f, +0.05617156f, +0.10911176f, +0.18613949f,
    -0.21304886f, -0.11837386f, -0.06452254f, +0.01450099f, -0.03497068f, +0.03907030f, +0.06927501f, +0.13114283f,
    -0.15195946f, -0.06528903f, +0.00816301f, +0.09342197f, -0.00768985f, +0.08454979f, -0.06193831f, +0.04520382f,
    -0.18858465f, -0.12311971f, -0.08049614f, +0.00820490f, -0.03343302f, +0.04559230f, +0.09504822f, +0.16720207f,
    -0.08559455f, -0.00763808f, -0.07567421f, +0.03534968f, -0.03516657f, +0.07333340f, +0.00215530f, +0.06659426f,
    -0.02403073f, +0.04535064f, +0.10581165f, +0.14817812f, -0.16961506f, -0.10086726f, -0.04851092f, +0.02657260f,
    -0.03184498f, +0.03237205f, +0.09189106f, +0.14247570f, -0.18240723f, -0.09515552f, +0.01455373f, +0.24037592f,
    -0.13847726f, -0.10706620f, -0.04225504f, +0.02279146f, -0.02027496f, +0.06288219f, +0.14652734f, +0.24736365f,
    -0.01184501f, +0.06392768f, +0.12518647f, +0.20364036f, -0.06881002f, -0.14446024f, -0.04796625f, +0.02247028f,
    -0.11420977f, -0.03750149f, +0.03140424f, +0.10375965f, -0.15867621f, -0.07792078f, -0.00786463f, +0.07086110f,
    -0.05512634f, +0.01544903f, +0.08794563f, +0.18253894f, -0.12583706f, -0.04047658f, +0.03500937f, +0.12212106f,
    -0.07983117f, -0.02346017f, +0.02269844f, +0.09270003f, -0.14228862f, -0.05948335f, +0.01340374f, +0.08643699f,
    -0.17088441f, -0.08146483f, +0.01637994f, +0.11269872f, -0.12229883f, -0.02740963f, +0.06919862f, +0.17516392f,
    -0.23416011f, -0.08861073f, -0.00531799f, +0.04334467f, -0.07542395f, -0.00959691f, +0.03128058f, +0.11384328f,
    -0.12321154f, -0.05411436f, -0.00802293f, +0.04527715f, -0.02979034f, +0.01261100f, +0.08631871f, +0.14489119f,
    -0.06713610f, -0.01768748f, +0.04439952f, +0.08539781f, -0.10447017f, -0.03861764f, +0.01176727f, +0.08397588f,
    -0.09664737f, -0.03306058f, +0.01965956f, +0.08313737f, -0.15701702f, -0.03552708f, +0.03436711f, +0.12348684f,
    -0.07465987f, +0.03148096f, -0.01592258f, +0.07807118f, -0.08365041f, -0.00777653f, +0.06189138f, +0.16461129f
};

// 2-bit TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis (k=2, L=8)
template<typename idx_t>
static __global__ void __launch_bounds__(256, 1) k_set_rows_turbo2_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turbo2_tcq * __restrict__ dst, const int64_t ne_total_groups,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int iq_is_k,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int grp = blockIdx.x;
    if (grp >= ne_total_groups) return;
    const int sid = threadIdx.x; // 0..255 = trellis state

    // Compute source and destination pointers (all threads, used by thread 0)
    const int64_t i_base = int64_t(grp) * QK_TURBO2_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turbo2_tcq * dst_blk = (block_turbo2_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                               + (i00 / QK_TURBO2_TCQ);

    __shared__ float x[128];
    __shared__ float cost[256];
    __shared__ uint8_t bt[128][128]; // 256 states, 4-bit packed (2 per byte), safe even/odd serialization
    __shared__ int warp_min_idx[8];
    __shared__ float warp_min_cost[8];
    __shared__ int shared_initial_state;

    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    if (d_innerq_calibrate && sid < 128) {
        atomicAdd(&d_innerq_channel_sq[sid], x[sid] * x[sid]);
        float abs_val = fabsf(x[sid]);
        unsigned int * addr = (unsigned int *)&d_innerq_channel_max[sid];
        unsigned int old_val = __float_as_uint(abs_val);
        unsigned int assumed;
        do {
            assumed = *addr;
            if (__uint_as_float(assumed) >= abs_val) break;
        } while (atomicCAS(addr, assumed, old_val) != assumed);
        if (sid == 0) atomicAdd(&d_innerq_count, 1);
    }
    if (sid < 128) x[sid] *= d_innerq_channel_scale[sid];
    __syncthreads();

    // Norm reduction
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // FWHT
    if (sid < 128) x[sid] *= d_turbo_wht_signs1[sid];
    __syncthreads();
    for (int h = 1; h < 128; h *= 2) {
        if (sid < 64) {
            int j = (sid / h) * (2 * h) + (sid % h);
            float a = x[j], b = x[j + h];
            x[j] = a + b; x[j + h] = a - b;
        }
        __syncthreads();
    }
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * d_turbo_wht_signs2[sid];
    __syncthreads();

    if (sid == 0) turbo_extract_append(x);
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Viterbi forward pass: free initial state (all equally viable)
    cost[sid] = 0.0f;
    __syncthreads();

    for (int t = 0; t < 128; t++) {
        float xt = x[t];

        // Right-shift trellis (k=2, L=8): ns = (prev >> 2) | (out << 6)
        // Predecessors of sid: prev = ((sid & 0x3F) << 2) | p, for p = 0..3
        int base_prev = (sid & 0x3F) << 2;
        float dist = xt - d_turbo2_tcq_codebook[sid];
        dist = dist * dist;

        float best = 1e30f;
        int best_p = 0;
        for (int p = 0; p < 4; p++) {
            float c = cost[base_prev | p];
            if (c < best) {
                best = c;
                best_p = p;
            }
        }

        __syncthreads();
        cost[sid] = best + dist;

        // Store backtrace: 4-bit packed, 2 entries per byte (safe even/odd serialization)
        if (sid % 2 == 0) {
            bt[t][sid / 2] = (uint8_t)(best_p & 0x3);
        }
        __syncthreads();
        if (sid % 2 == 1) {
            bt[t][sid / 2] |= ((uint8_t)(best_p & 0x3)) << 4;
        }
        __syncthreads();
    }

    // Warp argmin over 256 costs
    {
        float my_cost = cost[sid];
        int my_idx = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, my_cost, offset);
            int other_idx = __shfl_xor_sync(0xFFFFFFFF, my_idx, offset);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32] = my_idx;
        }
    }
    __syncthreads();
    if (sid == 0) {
        float best = warp_min_cost[0];
        int best_idx = warp_min_idx[0];
        for (int w = 1; w < 8; w++) {
            if (warp_min_cost[w] < best) { best = warp_min_cost[w]; best_idx = warp_min_idx[w]; }
        }
        shared_initial_state = best_idx; // temporarily: best final state (becomes initial after backtrack)
    }
    __syncthreads();

    // Save x[] to global buffer before backtrack overwrites it
    if (d_tcq_dump_max > 0 && grp < d_tcq_dump_max && sid < 128)
        d_tcq_dump_x_buf[grp * 128 + sid] = x[sid];

    // Backtrack (inherently sequential)
    uint8_t * outputs = (uint8_t *)x;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6);
            int p = (bt[t][state / 2] >> ((state % 2) * 4)) & 0x3;
            state = ((state & 0x3F) << 2) | p;
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // Save output symbols to global buffer
    if (d_tcq_dump_max > 0 && grp < d_tcq_dump_max && sid < 128)
        d_tcq_dump_out_buf[grp * 128 + sid] = outputs[sid];

    // Parallel recon norm: t>=3 can compute state directly from 4 outputs (4 shifts of 2 = 8 bits)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 3) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++)
                cur_state = (cur_state >> 2) | (((int)outputs[t]) << 6);
        } else {
            cur_state = ((int)outputs[sid - 3] & 0x3)
                      | (((int)outputs[sid - 2] & 0x3) << 2)
                      | (((int)outputs[sid - 1] & 0x3) << 4)
                      | (((int)outputs[sid]     & 0x3) << 6);
        }
        float c = d_turbo2_tcq_codebook[cur_state];
        my_recon_sq = c * c;
    }
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16);
        v += __shfl_down_sync(0xFFFFFFFF, v, 8);
        v += __shfl_down_sync(0xFFFFFFFF, v, 4);
        v += __shfl_down_sync(0xFFFFFFFF, v, 2);
        v += __shfl_down_sync(0xFFFFFFFF, v, 1);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= iq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Serial bitpack — byte-alignment prevents parallel atomicOr
    if (sid == 0) {
        for (int j = 0; j < 33; j++) dst_blk->qs[j] = 0;
        dst_blk->qs[0] = (uint8_t)((shared_initial_state >> 2) & 0x3F);
        for (int t = 0; t < 128; t++) {
            const int bit_pos = 6 + t * 2;
            const int byte_idx = bit_pos / 8;
            const int bit_off = bit_pos % 8;
            const int out = outputs[t] & 0x3;
            dst_blk->qs[byte_idx] |= (uint8_t)(out << bit_off);
        }
        dst_blk->norm = __float2half(corrected_norm);
    }
}

// 2-bit TCQ GET_ROWS dequantize
#define QR_TURBO2_TCQ 2
static __device__ __forceinline__
void dequantize_turbo2_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs: read 8-bit state via sliding window
    {
        const int t = iqs;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.x = d_turbo2_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64
    {
        const int t = iqs + 64;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.y = d_turbo2_tcq_codebook[state] * norm;
    }
}
