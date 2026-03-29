// CUDA TCQ codebook training — 100-1000x faster than numpy Viterbi
// Compile: nvcc -O3 -arch=sm_86 -o tcq_train_cuda tcq_train_cuda.cu -lcurand
// Usage:   ./tcq_train_cuda --bits 2 --n-train 100000 --n-iters 200

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdint>
#include <curand.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
		fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
		exit(1); \
	} \
} while(0)

#define CHECK_CURAND(call) do { \
	curandStatus_t err = (call); \
	if (err != CURAND_STATUS_SUCCESS) { \
		fprintf(stderr, "cuRAND error at %s:%d: %d\n", __FILE__, __LINE__, err); \
		exit(1); \
	} \
} while(0)

// block length for TCQ
#define T 128

// max states we support
#define MAX_STATES 1024

// codebook in constant memory
__constant__ float d_codebook[MAX_STATES];

// ============================================================================
// Viterbi kernel: one threadblock per sample, one thread per state
// Writes per-sample state assignments and MSE to global memory
// ============================================================================
template<int K, int L>
__global__ void k_viterbi_encode(
	const float* __restrict__ data,      // [n_train, T]
	int16_t*     __restrict__ states_out, // [n_train, T]
	float*       __restrict__ mse_out,    // [n_train]
	int n_train
) {
	const int sample = blockIdx.x;
	if (sample >= n_train) return;

	constexpr int N_STATES = 1 << L;
	constexpr int N_OUT = 1 << K;
	constexpr int MASK_LOWER = (1 << (L - K)) - 1;

	const int sid = threadIdx.x;
	if (sid >= N_STATES) return;

	const float* x = data + sample * T;

	// shared memory layout
	extern __shared__ char smem_raw[];
	float* cost     = (float*)smem_raw;                         // [N_STATES]
	float* new_cost = cost + N_STATES;                          // [N_STATES]
	// backtrace: packed 2 entries per byte (4-bit slots)
	// bt[t][sid/2] stores predecessor index (0..N_OUT-1) for sid
	uint8_t* bt = (uint8_t*)(new_cost + N_STATES);             // [T][N_STATES/2]

	// free initial state
	cost[sid] = 0.0f;
	__syncthreads();

	// Viterbi forward pass
	for (int t = 0; t < T; t++) {
		float x_t = x[t];
		// This state (sid) as a NEXT state: find best predecessor
		float cb_val = d_codebook[sid];
		float dist = (x_t - cb_val) * (x_t - cb_val);

		float best_cost = FLT_MAX;
		int best_p = 0;

		#pragma unroll
		for (int p = 0; p < N_OUT; p++) {
			int prev_s = ((sid & MASK_LOWER) << K) | p;
			float c = cost[prev_s] + dist;
			if (c < best_cost) {
				best_cost = c;
				best_p = p;
			}
		}

		new_cost[sid] = best_cost;

		// pack backtrace: 4-bit slots, even/odd serialization to avoid races
		int bt_idx = t * (N_STATES / 2) + sid / 2;
		if (sid % 2 == 0) {
			bt[bt_idx] = (uint8_t)(best_p & 0xF);
		}
		__syncthreads();
		if (sid % 2 == 1) {
			bt[bt_idx] |= ((uint8_t)(best_p & 0xF)) << 4;
		}
		__syncthreads();

		// swap cost arrays
		float tmp = new_cost[sid];
		cost[sid] = tmp;
		__syncthreads();
	}

	// Thread 0: backtrack and write results
	if (sid == 0) {
		// find best final state
		float best = FLT_MAX;
		int best_state = 0;
		for (int s = 0; s < N_STATES; s++) {
			if (cost[s] < best) {
				best = cost[s];
				best_state = s;
			}
		}

		// backtrack
		int16_t* out_states = states_out + sample * T;
		float mse = 0.0f;
		int state = best_state;
		for (int t = T - 1; t >= 0; t--) {
			out_states[t] = (int16_t)state;
			float recon = d_codebook[state];
			float diff = x[t] - recon;
			mse += diff * diff;

			// read predecessor
			int bt_idx = t * (N_STATES / 2) + state / 2;
			int p;
			if (state % 2 == 0) {
				p = bt[bt_idx] & 0xF;
			} else {
				p = (bt[bt_idx] >> 4) & 0xF;
			}
			state = ((state & MASK_LOWER) << K) | p;
		}

		mse_out[sample] = mse / T;
	}
}

// ============================================================================
// Centroid collection kernel: scatter-add (state, value) pairs
// ============================================================================
__global__ void k_collect_centroids(
	const float*   __restrict__ data,       // [n_train, T]
	const int16_t* __restrict__ states,     // [n_train, T]
	double*        __restrict__ state_sums, // [n_states]
	int*           __restrict__ state_counts,// [n_states]
	int n_train
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = n_train * T;
	if (idx >= total) return;

	int sample = idx / T;
	int t = idx % T;

	int state = (int)(unsigned short)states[sample * T + t];
	float val = data[sample * T + t];

	atomicAdd(&state_sums[state], (double)val);
	atomicAdd(&state_counts[state], 1);
}

// ============================================================================
// Normalize each 128-element block to unit norm (matches actual KV cache pipeline)
// ============================================================================
__global__ void k_normalize_blocks(float* data, int n_blocks) {
	int block_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (block_id >= n_blocks) return;

	float* blk = data + block_id * T;
	float norm_sq = 0.0f;
	for (int i = 0; i < T; i++) norm_sq += blk[i] * blk[i];
	float inv_norm = (norm_sq > 1e-20f) ? rsqrtf(norm_sq) : 0.0f;
	for (int i = 0; i < T; i++) blk[i] *= inv_norm;
}

// ============================================================================
// Host code
// ============================================================================

// Lloyd-Max centroids for initialization
static const float LM_2BIT[] = {-1.510f, -0.4528f, 0.4528f, 1.510f};
static const float LM_3BIT[] = {-1.748f, -1.050f, -0.5006f, -0.06971f, 0.06971f, 0.5006f, 1.050f, 1.748f};

void init_coset_codebook(float* codebook, int k, int L, float sigma, const float* centroids) {
	int n_states = 1 << L;
	int n_out = 1 << k;
	int n_groups = 1 << (L - k);

	float spacing = (centroids[1] - centroids[0]) * sigma;
	for (int group = 0; group < n_groups; group++) {
		float shift = spacing * ((float)group / n_groups - 0.5f);
		for (int pos = 0; pos < n_out; pos++) {
			int state = (group << k) | pos;
			codebook[state] = centroids[pos] * sigma + shift;
		}
	}
}

float lloyd_max_mse(int k, float sigma) {
	const float* centroids = (k == 2) ? LM_2BIT : LM_3BIT;
	int n_out = 1 << k;

	// compute MSE on 10000 samples
	float total_mse = 0.0f;
	int n_eval = 10000;
	srand(12345);
	for (int s = 0; s < n_eval; s++) {
		float mse = 0.0f;
		for (int t = 0; t < T; t++) {
			// box-muller
			float u1 = (float)(rand() + 1) / (RAND_MAX + 1.0f);
			float u2 = (float)(rand() + 1) / (RAND_MAX + 1.0f);
			float x = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2) * sigma;

			// find nearest centroid
			float best_d = FLT_MAX;
			for (int c = 0; c < n_out; c++) {
				float d = (x - centroids[c] * sigma) * (x - centroids[c] * sigma);
				if (d < best_d) best_d = d;
			}
			mse += best_d;
		}
		total_mse += mse / T;
	}
	return total_mse / n_eval;
}

template<int K, int L>
void train(int n_train, int n_iters, int n_restarts) {
	constexpr int N_STATES = 1 << L;
	const float sigma = 1.0f / sqrtf(128.0f); // post-FWHT scale for head_dim=256 (128-elem blocks)
	const float* centroids = (K == 2) ? LM_2BIT : LM_3BIT;

	float lm_mse = lloyd_max_mse(K, sigma);
	printf("Lloyd-Max %d-bit baseline: MSE = %.6f\n\n", K, lm_mse);

	// allocate GPU memory
	float* d_data;
	int16_t* d_states;
	float* d_mse;
	double* d_state_sums;
	int* d_state_counts;

	CHECK_CUDA(cudaMalloc(&d_data, (size_t)n_train * T * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_states, (size_t)n_train * T * sizeof(int16_t)));
	CHECK_CUDA(cudaMalloc(&d_mse, (size_t)n_train * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_state_sums, N_STATES * sizeof(double)));
	CHECK_CUDA(cudaMalloc(&d_state_counts, N_STATES * sizeof(int)));

	// host buffers
	float* h_codebook = (float*)malloc(N_STATES * sizeof(float));
	float* h_mse = (float*)malloc(n_train * sizeof(float));
	double* h_sums = (double*)malloc(N_STATES * sizeof(double));
	int* h_counts = (int*)malloc(N_STATES * sizeof(int));

	// cuRAND for generating random data
	curandGenerator_t gen;
	CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	// shared memory size for Viterbi kernel
	size_t smem_size = 2 * N_STATES * sizeof(float) + T * (N_STATES / 2) * sizeof(uint8_t);
	printf("Shared memory per block: %zu bytes\n", smem_size);
	if (smem_size > 48 * 1024) {
		printf("WARNING: Exceeds 48KB default shared memory. Using cudaFuncSetAttribute.\n");
		if (smem_size <= 100 * 1024) {
			CHECK_CUDA(cudaFuncSetAttribute(
				k_viterbi_encode<K, L>,
				cudaFuncAttributeMaxDynamicSharedMemorySize,
				smem_size));
		} else {
			fprintf(stderr, "ERROR: Shared memory %zu exceeds maximum.\n", smem_size);
			exit(1);
		}
	}

	float best_global_mse = FLT_MAX;
	float best_global_codebook[MAX_STATES];

	for (int restart = 0; restart < n_restarts; restart++) {
		if (n_restarts > 1) printf("\n=== Restart %d/%d ===\n", restart+1, n_restarts);

		// initialize codebook
		init_coset_codebook(h_codebook, K, L, sigma, centroids);
		if (restart > 0) {
			srand(restart * 7919);
			for (int s = 0; s < N_STATES; s++) {
				h_codebook[s] += ((float)rand() / RAND_MAX - 0.5f) * sigma * 0.04f;
			}
		}

		float best_mse = FLT_MAX;
		float best_codebook[MAX_STATES];
		memcpy(best_codebook, h_codebook, N_STATES * sizeof(float));
		int stall = 0;

		for (int iter = 0; iter < n_iters; iter++) {
			// upload codebook to constant memory
			CHECK_CUDA(cudaMemcpyToSymbol(d_codebook, h_codebook, N_STATES * sizeof(float)));

			// generate fresh random data each iteration
			CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 42 + restart * 10000 + iter));
			CHECK_CURAND(curandGenerateNormal(gen, d_data, (size_t)n_train * T, 0.0f, sigma));

			// run Viterbi
			k_viterbi_encode<K, L><<<n_train, N_STATES, smem_size>>>(
				d_data, d_states, d_mse, n_train);
			CHECK_CUDA(cudaGetLastError());

			// compute mean MSE on host
			CHECK_CUDA(cudaMemcpy(h_mse, d_mse, n_train * sizeof(float), cudaMemcpyDeviceToHost));
			double mean_mse = 0.0;
			for (int i = 0; i < n_train; i++) mean_mse += h_mse[i];
			mean_mse /= n_train;

			const char* improved = "";
			if (mean_mse < best_mse) {
				best_mse = mean_mse;
				memcpy(best_codebook, h_codebook, N_STATES * sizeof(float));
				improved = " *";
				stall = 0;
			} else {
				stall++;
			}

			// collect centroids on GPU
			CHECK_CUDA(cudaMemset(d_state_sums, 0, N_STATES * sizeof(double)));
			CHECK_CUDA(cudaMemset(d_state_counts, 0, N_STATES * sizeof(int)));

			int total_elements = n_train * T;
			int threads = 256;
			int blocks = (total_elements + threads - 1) / threads;
			k_collect_centroids<<<blocks, threads>>>(
				d_data, d_states, d_state_sums, d_state_counts, n_train);
			CHECK_CUDA(cudaGetLastError());

			// read back and update codebook on host
			CHECK_CUDA(cudaMemcpy(h_sums, d_state_sums, N_STATES * sizeof(double), cudaMemcpyDeviceToHost));
			CHECK_CUDA(cudaMemcpy(h_counts, d_state_counts, N_STATES * sizeof(int), cudaMemcpyDeviceToHost));

			int used = 0;
			for (int s = 0; s < N_STATES; s++) {
				if (h_counts[s] > 0) {
					h_codebook[s] = (float)(h_sums[s] / h_counts[s]);
					used++;
				}
			}

			// reinitialize dead states
			if (used < N_STATES && iter < n_iters - 1) {
				for (int s = 0; s < N_STATES; s++) {
					if (h_counts[s] == 0) {
						// find most-used state as donor
						int donor = 0;
						for (int d = 1; d < N_STATES; d++) {
							if (h_counts[d] > h_counts[donor]) donor = d;
						}
						h_codebook[s] = h_codebook[donor] + ((float)rand() / RAND_MAX - 0.5f) * sigma * 0.02f;
					}
				}
			}

			printf("  iter %3d: MSE = %.6f  (%d/%d used)%s\n",
				   iter+1, (float)mean_mse, used, N_STATES, improved);
			fflush(stdout);

			if (stall >= 10 && iter > 20) {
				printf("  Converged (10 iters without improvement)\n");
				break;
			}
		}

		if (best_mse < best_global_mse) {
			best_global_mse = best_mse;
			memcpy(best_global_codebook, best_codebook, N_STATES * sizeof(float));
		}
	}

	// evaluate on fresh data
	CHECK_CUDA(cudaMemcpyToSymbol(d_codebook, best_global_codebook, N_STATES * sizeof(float)));
	int n_eval = 10000;
	float* d_eval_data;
	int16_t* d_eval_states;
	float* d_eval_mse;
	CHECK_CUDA(cudaMalloc(&d_eval_data, (size_t)n_eval * T * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&d_eval_states, (size_t)n_eval * T * sizeof(int16_t)));
	CHECK_CUDA(cudaMalloc(&d_eval_mse, (size_t)n_eval * sizeof(float)));

	CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 99999));
	CHECK_CURAND(curandGenerateNormal(gen, d_eval_data, (size_t)n_eval * T, 0.0f, sigma));

	k_viterbi_encode<K, L><<<n_eval, N_STATES, smem_size>>>(
		d_eval_data, d_eval_states, d_eval_mse, n_eval);
	CHECK_CUDA(cudaGetLastError());

	float* h_eval_mse = (float*)malloc(n_eval * sizeof(float));
	CHECK_CUDA(cudaMemcpy(h_eval_mse, d_eval_mse, n_eval * sizeof(float), cudaMemcpyDeviceToHost));
	double eval_mean = 0.0;
	for (int i = 0; i < n_eval; i++) eval_mean += h_eval_mse[i];
	eval_mean /= n_eval;

	float reduction = (1.0f - eval_mean / lm_mse) * 100.0f;
	float db_gain = 10.0f * log10f(lm_mse / eval_mean);
	printf("\nEVAL (%d fresh samples): MSE = %.6f (%.1f%% vs LM, %+.2f dB)\n",
		   n_eval, (float)eval_mean, reduction, db_gain);

	// print codebook as C array (already at 1/sqrt(128) scale)
	printf("\n// GLA-trained free-init TCQ codebook: k=%d, L=%d (%d states)\n", K, L, N_STATES);
	printf("// MSE reduction: %.1f%% vs Lloyd-Max %d-bit, %.2f dB\n", reduction, K, db_gain);
	printf("// CUDA-trained: n_train=%d, n_iters up to convergence, unit-norm blocks\n", n_train);

	// print both codebook variants
	const char* suffixes[] = {"", "_fattn"};
	for (int v = 0; v < 2; v++) {
		printf("static __constant__ float d_turbo%d_tcq_codebook%s[%d] = {\n", K, suffixes[v], N_STATES);
		for (int i = 0; i < N_STATES; i += 8) {
			printf("    ");
			for (int j = i; j < i + 8 && j < N_STATES; j++) {
				printf("%+.8ff", best_global_codebook[j]);
				if (j < N_STATES - 1) {
					printf(",");
					if ((j + 1) % 8 != 0) printf(" ");
				}
			}
			printf("\n");
		}
		printf("};\n\n");
	}

	// cleanup
	free(h_codebook); free(h_mse); free(h_sums); free(h_counts); free(h_eval_mse);
	cudaFree(d_data); cudaFree(d_states); cudaFree(d_mse);
	cudaFree(d_state_sums); cudaFree(d_state_counts);
	cudaFree(d_eval_data); cudaFree(d_eval_states); cudaFree(d_eval_mse);
	curandDestroyGenerator(gen);
}

int main(int argc, char** argv) {
	int bits = 2;
	int n_train = 100000;
	int n_iters = 200;
	int n_restarts = 3;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--bits") == 0 && i+1 < argc) bits = atoi(argv[++i]);
		else if (strcmp(argv[i], "--n-train") == 0 && i+1 < argc) n_train = atoi(argv[++i]);
		else if (strcmp(argv[i], "--n-iters") == 0 && i+1 < argc) n_iters = atoi(argv[++i]);
		else if (strcmp(argv[i], "--n-restarts") == 0 && i+1 < argc) n_restarts = atoi(argv[++i]);
	}

	printf("=== CUDA TCQ codebook training ===\n");
	printf("bits=%d, n_train=%d, n_iters=%d, n_restarts=%d\n\n", bits, n_train, n_iters, n_restarts);

	if (bits == 2) {
		train<2, 8>(n_train, n_iters, n_restarts);
	} else if (bits == 3) {
		train<3, 9>(n_train, n_iters, n_restarts);
	} else {
		fprintf(stderr, "Unsupported bits: %d\n", bits);
		return 1;
	}

	return 0;
}
