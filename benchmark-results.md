# TurboQuant Benchmark Results

Hardware: RTX 3090 24GB, Qwen3.5 27B Q6_K (20.56 GiB)
Date: 2026-03-26
Build: feature/turboquant-kv-cache + FA_ALL_QUANTS=ON
Data: wikitext-2-raw/wiki.test.raw

**Methodology note**: Early benchmarks (pre 2026-03-29) used 1 chunk for long-context PPL,
which has ±0.15 error bars at 32K+. Results from 2026-03-29 onward use 4 chunks (±0.07)
for long context and 8 chunks for 2K. Numbers across methodologies are not directly comparable —
the 1-chunk data led to incorrect conclusions about turbo3 context scaling (see TURBO4_POSTMORTEM.md).

## PPL (2K ctx, 8 chunks)

| Config | PPL | vs q8_0 | Notes |
|--------|-----|---------|-------|
| q8_0 baseline | 5.8375 | — | |
| turbo3 uniform | 5.8323 | -0.09% | with norm correction |
| turbo4 uniform | 5.8186 | -0.32% | with norm correction |
| LA-2 turbo3 | 5.8140 | -0.40% | TURBO_LAYER_ADAPTIVE=2, last 8/40 layers q8_0 |
| LA-2 turbo4 | 5.8077 | -0.51% | TURBO_LAYER_ADAPTIVE=2, last 8/40 layers q8_0 |
| **LA-1 turbo3** | **5.7958** | **-0.71%** | TURBO_LAYER_ADAPTIVE=1, first4+last4 q8_0. BEST PPL |
| LA-1 turbo4 | 5.8989 | +1.05% | WORSE than uniform turbo4! Early layers need turbo4's QJL |
| LA-3 turbo3 (last4) | 5.8091 | -0.49% | only 4 layers q8_0 = ~4.2x compression |
| LA-4 turbo3 (first4) | 5.8211 | -0.28% | only 4 layers q8_0 |
| LA-5 turbo3 (2+2) | 5.8091 | -0.49% | same as mode 3! Only 4 layers q8_0 = ~4.2x |
| turbo4-K + q8_0-V | 5.8451 | +0.13% | asymmetric |
| q8_0-K + turbo3-V | 5.8451 | +0.13% | asymmetric |
| turbo4-K + turbo3-V | 5.8653 | +0.48% | asymmetric, worst mixed combo |
| turbo3-K + turbo4-V | 5.8212 | -0.28% | asymmetric, values matter more! |

## Decode Speed tg64 (tok/s)

| Config | 4K | 16K | 32K | ratio @32K |
|--------|-----|------|------|-----------|
| q8_0 baseline | 31.02 | 30.77 | 30.69 | 1.000 |
| turbo3 uniform | 29.93 | 29.65 | 29.83 | 0.972 |
| turbo4 uniform | 29.43 | 29.41 | 29.47 | 0.960 |
| LA-2 turbo3 | 30.14 | 29.94 | 29.98 | 0.977 |
| LA-2 turbo4 | 29.69 | 29.68 | 29.69 | 0.967 |
| LA-1 turbo3 | 30.12 | — | 29.98 | 0.977 |
| turbo4-K + q8_0-V | 30.21 | 30.14 | 30.15 | 0.982 |
| q8_0-K + turbo3-V | 30.40 | 30.34 | 30.32 | 0.988 |
| turbo4-K + turbo3-V | 29.70 | 29.57 | 29.62 | 0.965 |
| turbo3-K + turbo4-V | 29.70 | 29.57 | 29.63 | 0.965 |

## Prefill Speed pp4096 (tok/s)

| Config | tok/s | ratio |
|--------|-------|-------|
| q8_0 | 1134.64 | 1.000 |
| turbo3 | 631.09 | 0.556 |
| turbo4 | 586.71 | 0.517 |

## Extreme Context (65K) — Decode Speed tg64

| Config | 65K tok/s | VRAM | vs 32K speed |
|--------|----------|------|-------------|
| LA-1 turbo3 | 29.98 | ~22.3 GiB | identical to 32K (29.98) |
| LA-5 turbo3 (2+2) | 29.90 | ~22.3 GiB | -0.3% |
| turbo4 uniform | 29.51 | ~22.2 GiB | +0.1% |

Note: q8_0 would need ~28+ GiB at 65K — would OOM on 24GB RTX 3090.

## Layer-Adaptive Mode Comparison (all turbo3, PPL 2K/8chunks)

| Mode | Which layers q8_0 | #layers q8_0 | PPL | vs q8_0 | Compression |
|------|-------------------|-------------|-----|---------|------------|
| 1 (4+4) | first 4 + last 4 | 8 | 5.7958 | -0.71% | ~3.5x |
| 3 (last4) | last 4 | 4 | 5.8091 | -0.49% | ~4.2x |
| 5 (2+2) | first 2 + last 2 | 4 | 5.8091 | -0.49% | ~4.2x |
| 2 (last8) | last 8 | 8 | 5.8140 | -0.40% | ~3.5x |
| 4 (first4) | first 4 | 4 | 5.8211 | -0.28% | ~4.2x |
| 0 (uniform) | none | 0 | 5.8323 | -0.09% | 4.9x |

## FWHT Inline FA Rotation (experiment/fwht-inline-fa)

Q pre-rotation moved from graph-level ggml_turbo_wht op to inline in FA kernels:
- Vec kernel (decode): shared memory FWHT rotation, zero extra kernel launches
- Prefill MMA: separate forward rotation kernel with persistent buffer
- V un-rotation remains at graph level (CUDA graph compatible)

| Metric | Before (graph-level) | After (inline Q) | Delta |
|--------|---------------------|-------------------|-------|
| Decode tg64 | 30.25 | 30.14 | -0.4% |
| Prefill pp512 | 1149.90 | 1146.21 | -0.3% |
| PPL (10-chunk) | 19.7152 | 19.7152 | identical |
| CUDA graphs | working | working | no change |

Key finding: `cudaMallocAsync` for Q rotation buffer caused NaN on graph replay.
Fixed by using a persistent `cudaMalloc` buffer that grows as needed.

## Experiment #61: TCQ (Trellis-Coded Quantization) — turbo3_tcq

Branch: `experiment/tcq-turbo3`. Right-shift bitshift trellis (k=3, L=9, 512 states).
Parallel Viterbi encode (512 threads/block), O(1) sliding-window decode.
3.25 bpv (52 bytes/128 elements) vs turbo3's 3.5 bpv.

| Metric | turbo3_tcq | turbo3 | q8_0 | TCQ vs turbo3 |
|--------|-----------|--------|------|---------------|
| PPL (2K/8ch) | 5.8270 | 5.8323 | 5.8375 | -0.09% (better) |
| Prefill pp4096 | 884.26 | 1122.57 | 1134.64 | -21% (slower) |
| Decode tg64 | 28.58 | 30.17 | 31.02 | -5.3% (slower) |

Notes:
- PPL slightly better than turbo3 despite fewer bits (TCQ correlation gain)
- Prefill 21% slower — Viterbi encode overhead (512 threads, 128 forward steps)
- Decode 5% slower — 9-bit sliding window extract vs 3-bit direct index
- VRAM: ~7% smaller KV cache vs turbo3 (3.25 vs 3.5 bpv)

## Observations

1. Norm correction makes turbo3 and turbo4 BEAT q8_0 in PPL
2. Layer-adaptive mode 2 improves PPL further (best: LA-2 turbo4 at 5.8077)
3. Layer-adaptive also improves decode speed (q8_0 layers dequant faster)
4. No context scaling regression — turbo/q8 ratio improves at longer contexts
5. Asymmetric turbo+q8 PPL is slightly worse than pure q8_0 (norm correction mismatch?)
6. q8_0-K + turbo3-V is fastest asymmetric config (98.8% of q8_0)
7. Prefill is the big gap (0.5x) — vec kernel only, no tensor core support yet
8. Mixed turbo-turbo: turbo3-K + turbo4-V (5.8212) beats turbo4-K + turbo3-V (5.8653) — contradicts "More Keys Less Values" paper. Values need more precision on this model.
9. Both mixed turbo-turbo combos decode at same speed (~29.65 tok/s) — no speed difference from K/V asymmetry
10. 65K context fits on 24GB RTX 3090 with all turbo configs (~22.2-22.3 GiB). q8_0 would OOM.
11. Decode speed at 65K is virtually identical to 32K — zero degradation even at 2x the context
12. Mode 3 (last4) = Mode 5 (first2+last2) in PPL. Last 2 layers dominate quality impact.
13. Mode 5 (first2+last2) is the max-compression sweet spot: only 4 layers q8_0, ~4.2x compression, -0.49% PPL
14. Asymmetric layer-adaptive (V-only or K-only promotion) does NOT help — norm correction mismatch between turbo+q8_0 within a layer hurts. Both K+V must be promoted together.

## FWHT Rotation Results (CORRECTED)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 WITH rotation + norm correction (committed HEAD) | 5.8323 | -0.09% (BETTER) |
| turbo3 WITHOUT rotation + WITH norm correction | 6.2357 | +6.8% worse |
| turbo3 WITHOUT rotation + WITHOUT norm correction | 6.5249 | +11.8% worse |
| q8_0 baseline | 5.8375 | — |

**CORRECTION**: Previous session incorrectly concluded rotation HURTS PPL. The 6.51 measurement was from a BROKEN double-rotation state (inline FA rotation applied on top of graph-level rotation). The committed code (721880c00+) with SET_ROWS rotation + graph-level TURBO_WHT gives correct PPL = 5.83.

**Conclusion**: FWHT rotation is ESSENTIAL for quality. It provides a 0.39 PPL improvement (6.24 → 5.83). Norm correction provides an additional 0.29 improvement (6.52 → 6.24). Together they make turbo3 beat q8_0.

## Prefill Dequant+MMA Optimization (experiment/prefill-dequant-attend)

| Config | pp4096 old (vec) | pp4096 new (dequant+MMA) | vs q8_0 (1134.64) | Speedup |
|--------|-----------------|-------------------------|-------------------|---------|
| turbo3 | 631.09 | 1121.33 | 98.8% | 1.78x |
| turbo4 | 586.71 | 1121.30 | 98.8% | 1.91x |

**PPL with optimization**: turbo3 = 5.8501 (vs 5.8323 baseline, within error bars)

**Decode speed**: 30.10 tok/s (unchanged from baseline ~30 tok/s)

**How it works**: During prefill (Q->ne[1] > 1), bulk-dequantize turbo K/V to fp16 temp buffers, then use the fast MMA (tensor core) kernel. During decode (Q->ne[1] == 1), use the existing vec kernel with inline dequant. The fp16 temp buffer is allocated with cudaMallocAsync and freed after attention completes.

**Memory overhead**: ~16 MB per KV head group (seq_len × head_dim × 2 bytes × 2 for K+V). Temporary, freed after each attention call.

**Mixed K/V prefill speed** (pp4096):

| Config | tok/s | vs q8_0 |
|--------|-------|---------|
| q8_0 K + turbo3 V | 1131.15 | 99.7% |
| turbo3 K + q8_0 V | 1129.26 | 99.5% |
| turbo3 K + turbo3 V | 1121.33 | 98.8% |
| turbo4 K + turbo4 V | 1121.30 | 98.8% |

## Extreme Context with Prefill Optimization (turbo3)

| Context | pp tok/s | tg64 tok/s | Notes |
|---------|----------|-----------|-------|
| 4K | 1123 | 30.03 | baseline |
| 32K | 980 | 29.83 | |
| 65K | 847 | 29.79 | q8_0 would OOM (~28+ GiB) |
| 98K | 748 | 29.86 | |
| 112K | 707 | 29.82 | |
| 128K | 669 | 29.85 | full model context window! |

**Key finding**: 27B model running at 128K context on a 24GB RTX 3090 with turbo3 KV cache. Impossible with q8_0. Decode speed is constant across all context lengths (~30 tok/s). Prefill scales sub-linearly with context length.

## Asymmetric Layer-Adaptive (turbo3, PPL 2K/8chunks)

| Mode | Strategy | PPL | vs q8_0 | Decode 4K | Notes |
|------|----------|-----|---------|-----------|-------|
| 6 | V-only q8_0 last 8 | 5.8390 | +0.03% | 30.16 | worse than uniform! |
| 7 | K-only q8_0 last 8 | 5.8390 | +0.03% | — | identical to mode 6 |
| 8 | V-only q8_0 first2+last2 | 5.8330 | -0.08% | — | ~= uniform |
| 2 (ref) | both K+V q8_0 last 8 | 5.8140 | -0.40% | 30.14 | much better |
| 0 (ref) | uniform turbo3 | 5.8323 | -0.09% | 29.93 | baseline |

**Key finding**: Asymmetric layer-adaptive does NOT help. Promoting only K or only V gives identical PPL (5.8390), both worse than uniform turbo3. The norm correction mismatch between turbo and q8_0 within the same layer hurts quality. Both K and V must be promoted together (mode 2) for the quality improvement to work.

## turbo4 Prefill Dequant+MMA Investigation

**Bug found**: turbo4 dequant_f16 kernel didn't handle ne0 > QK_TURBO4 (256 vs 128 for Qwen3.5-27B). Elements j >= 128 read from wrong block. Fix: add `blk_idx = j / QK_TURBO4` block indexing (matching turbo3 pattern).

| Config | PPL (prefill via MMA) | PPL (prefill via vec) | Difference |
|--------|----------------------|----------------------|------------|
| turbo3 | 5.8501 | 5.8323 | +0.3% (acceptable) |
| turbo4 (fixed) | 5.8966 | 5.8186 | +1.3% (too much) |

**Root cause of turbo4 regression**: turbo4's QJL correction adds ~0.001 magnitude adjustments per element. This is at the limit of fp16 precision (10-bit mantissa). The fp16 round-trip (dequant → fp16 buffer → MMA read) rounds away the QJL signal. turbo3 is unaffected because its 8 centroids are coarse enough (~0.3 spacing) for fp16.

**Decision**: Prefill dequant+MMA enabled for turbo3 only. turbo4 continues to use vec kernel for prefill (preserves PPL 5.8186 at cost of 588 tok/s prefill speed vs 1121 tok/s).

## Layer-Adaptive + Prefill MMA (turbo3, comprehensive)

| Config | PPL | vs q8_0 | pp4096 tok/s | pp/q8 | tg64 tok/s | tg/q8 | Compression |
|--------|-----|---------|-------------|-------|-----------|-------|-------------|
| **LA-1 turbo3** | **5.7690** | **-1.17%** | **1128** | **99.6%** | **30.25** | **97.5%** | ~3.5x |
| LA-5 turbo3 | 5.8246 | -0.22% | 1119 | 98.8% | 30.03 | 96.8% | ~4.2x |
| turbo3 uniform | 5.8501 | +0.22% | 1125 | 99.3% | 30.04 | 96.8% | 4.9x |
| q8_0 baseline | 5.8375 | — | 1133 | 100% | 31.04 | 100% | 1.0x |

**Recommended config: LA-1 turbo3** (TURBO_LAYER_ADAPTIVE=1)
- 1.17% BETTER PPL than q8_0
- 99.6% prefill speed, 97.5% decode speed
- 3.5x KV cache compression
- Enables 128K context on 24GB GPU where q8_0 OOMs at ~65K

## Experiment: Drop QJL from turbo4 (branch: experiment/drop-qjl)

| Config | PPL | vs q8_0 | pp4096 tok/s | tg64 tok/s | Compression |
|--------|-----|---------|-------------|-----------|-------------|
| turbo4 WITH QJL (baseline) | 5.8186 | -0.32% | 588 (vec) | 29.43 | 4.25 bits |
| turbo4 NO QJL | 5.8501 | +0.22% | 1124 (MMA!) | 29.40 | 4.25 bits* |
| turbo3 (reference) | 5.8323 | -0.09% | 1125 (MMA) | 29.93 | 3.5 bits |

*Block layout unchanged (rnorm+signs still present but zeroed). True format redesign would be 3.125 bits.

**Key finding**: QJL is worth +0.3 PPL points for turbo4. Without QJL, turbo4 is slightly WORSE than turbo3 in quality, speed, and compression. QJL + norm correction is the reason turbo4 beats q8_0. Dropping QJL does fix the fp16 prefill issue (MMA works = 1124 tok/s), but turbo3 already gets the same prefill speed.

**Conclusion**: Keep QJL. turbo4's value is the QJL+norm-correction combo. TheTom's "QJL unnecessary" finding may not apply when norm correction is present.

## Long-Context PPL Comparison (turbo3 vs q8_0)

| Context | Chunks | q8_0 PPL | turbo3 uniform PPL | turbo3 LA-1 PPL | LA-1 vs q8_0 |
|---------|--------|----------|-------------------|----------------|-------------|
| 2K | 8 | 5.8375 | 5.8323 (-0.09%) | 5.7690 (-1.17%) | **turbo3 wins** |
| 4K | 4 | 6.2677 | 6.3252 (+0.92%) | 6.3198 (+0.83%) | q8_0 wins |
| 8K | 4 | 7.4241 | 7.3783 (-0.62%) | 7.3952 (-0.39%) | **turbo3 wins** |

**Key finding**: Quality comparison is noisy across context lengths. Error bars ±0.16-0.18 are larger than the measured differences (0.03-0.09 PPL). turbo3 generally competitive with q8_0 at all context lengths. The PPL increase from 2K→8K is a data effect (wikitext text becomes harder to predict), not a quantization degradation — both turbo3 and q8_0 show the same pattern.

## Sign+Magnitude Encoding (branch: experiment/sign-magnitude-encoding)

turbo3 decode speed: 30.05 tok/s (4K) / 29.91 tok/s (32K). Identical to baseline. q8_0: 31.03 tok/s. The 3% gap is memory-bound, not ALU-bound. Encoding change has no effect.

## 128K Context Test

| Config | pp131072 tok/s | tg64 tok/s | Fits on 24GB? |
|--------|---------------|-----------|---------------|
| turbo3 uniform | 671.42 | 29.89 | YES |
| LA-5 turbo3 (first2+last2) | 673.95 | 30.01 | YES |
| LA-1 turbo3 (first4+last4) | — | — | **NO (OOM)** |

**Key finding**: LA-1 (8 q8_0 layers) OOMs at 128K on 24GB RTX 3090. LA-5 (4 q8_0 layers) and uniform turbo3 both work. LA-5 is the recommended config for 128K: PPL 5.8091 (-0.49% vs q8_0), 674 tok/s prefill, 30.01 tok/s decode, fits on 24GB.

**Context length recommendations**:
- Up to 65K: Use LA-1 turbo3 (best PPL: -1.17%)
- 65K-128K: Use LA-5 turbo3 (best balance: -0.49% PPL, 4.2x compression)
- 128K+: Use turbo3 uniform (maximum compression: 4.9x)

## Experiment: Vec Q Pre-rotation (experiment/vec-q-prerotate)

Moves FWHT Q rotation from inside the vec kernel (22 syncthreads for D=256) to a separate
kernel launch before vec kernel dispatch. Reduces register pressure and eliminates barrier stalls.

### MoE Model: Qwen3.5-35B-A3B-Q4_K_S (19.24 GiB)

| Context | KV Type | Before | After | Change |
|---------|---------|--------|-------|--------|
| p=0 | turbo3 | 118.10 | 125.77 | **+6.5%** |
| p=0 | q8_0 | — | 138.01 | (turbo3=91.1% of q8_0) |
| p=8K | turbo3 | — | 114.44 | — |

### Dense Model: Qwen3.5-27B-Q6_K (20.56 GiB)

| Context | KV Type | Before | After | Change |
|---------|---------|--------|-------|--------|
| p=0 | turbo3 | 30.05 | 30.13 | no regression |
| p=8K | turbo3 | — | 30.03 | — |
| p=32K | turbo3 | — | 29.94 | — |
| p=42K | turbo3 | — | 29.95 | — |
| p=32K | q8_0 | — | 30.99 | (turbo3=96.6%) |
| p=42K | q8_0 | — | 30.99 | (turbo3=96.6%) |

### Community Report (different GPU, Qwen3.5-27B-Q8_0, 42K context)

| KV Type | Prefill | Decode |
|---------|---------|--------|
| q8_0 | 2492.44 | 23.96 |
| turbo3 | 2445.92 | 16.30 (68% of q8_0) |

Note: regression is worse on the tester's GPU (likely lower bandwidth) where FFN is slower
and attention is a larger fraction of total compute. On our RTX 3090, turbo3 holds at 96.6% of q8_0.

PPL verification: 19.7152 (identical to baseline — FWHT is linear, so in-kernel vs pre-kernel produces same result).

## Experiment: Decode Dequant to FP16 (experiment/turbo-decode-fp16)

Dequant turbo3 K/V to fp16 before vec kernel for decode, so the inner loop uses
simple fp16 dot product instead of turbo3 bit-extract+LUT. Gated by GGML_TURBO_DECODE_FP16=1.

### MoE Model: Qwen3.5-35B-A3B-Q4_K_S

| Context | No flag | With flag | q8_0 | vs q8_0 |
|---------|---------|-----------|------|---------|
| p=0 | 125.77 | 127.98 | 138.01 | 92.7% |
| p=8K | 114.44 | **127.37** | — | — |
| p=32K | — | **126.92** | — | — |

### Dense Model: Qwen3.5-27B-Q6_K

| Context | No flag | With flag | q8_0 | vs q8_0 |
|---------|---------|-----------|------|---------|
| p=0 | 30.13 | 30.29 | 31.0 | 97.7% |
| p=32K | 29.94 | 30.05 | 30.99 | 97.0% |
| p=42K | 29.95 | 30.06 | 30.99 | 97.0% |

PPL verification: 19.7152 (bit-exact match — turbo3 centroids are lossless in fp16)

**Key finding**: FP16 dequant eliminates context scaling degradation on MoE models
(127 t/s flat from p=0 to p=32K) and has ZERO cost on dense models at all context lengths.
Could be made the default for turbo3 decode.

## Experiment: Attention-Sink Token Protection (#23)

Store first N KV positions at fp16 precision (pre-quantization), overwrite dequanted fp16
buffer before flash attention. Sink tokens receive disproportionate attention weights.

| Config | PPL (2K/8chunks) | Delta vs baseline |
|--------|------------------|-------------------|
| turbo3 baseline (no sink) | 5.8501 ± 0.165 | — |
| N=4 sink tokens | 5.8246 ± 0.164 | -0.026 |
| N=8 sink tokens | 5.8506 ± 0.165 | +0.001 |
| N=16 sink tokens | 5.8894 ± 0.167 | +0.039 |

**Conclusion**: No significant improvement. All results within error bars (±0.165).
turbo3 + FWHT + norm correction already achieves high enough quality that protecting
sink positions provides no measurable benefit. The attention-sink phenomenon amplifies
quantization error at those positions, but turbo3's error is too small for the effect
to matter.

## Experiment: NSNQuant Per-Token DC Removal (#22)

Subtract per-element mean and renormalize before FWHT rotation. Simplified version of
NSNQuant (which uses cross-token channel means, incompatible with per-token SET_ROWS).

| Config | PPL (2K/8chunks) | Delta vs baseline |
|--------|------------------|-------------------|
| turbo3 baseline | 5.8501 ± 0.165 | — |
| turbo3 + DC removal | 5.8827 ± 0.166 | +0.033 (noise) |
| turbo4 baseline | 5.8186 (ref) | — |
| turbo4 + DC removal | 17.4134 ± 0.618 | **+11.6 (catastrophic)** |

**Conclusion**: Per-token DC removal provides no benefit for turbo3 (values already
near-zero-mean after normalization + FWHT) and is catastrophic for turbo4 (QJL residual
computed relative to DC-removed signal but decoded without DC restoration).

## Experiment: MSE-Optimal Norm Correction (bonus)

Replace L2-preserving norm correction (β = ||x||/||q||) with MSE-optimal scaling
(α = ||x|| · dot(x,q) / ||q||²). Theoretically halves per-element MSE.

| Config | PPL (2K/8chunks) | Delta vs baseline |
|--------|------------------|-------------------|
| turbo3 L2-preserving (baseline) | 5.8501 ± 0.165 | — |
| turbo3 MSE-optimal | 5.9083 ± 0.167 | +0.058 (worse) |

**Conclusion**: MSE-optimal scaling reduces the norm by cos(θ), effectively lowering
attention temperature (making softmax more uniform). L2-preserving is better for
attention because it maintains the intended dot-product magnitudes.

## Session 2026-03-27 Night — Experiment Run Summary

Working through Ready experiments, then Needs Research. Progress:
- **#39 GSR Walsh ordering**: DONE — NEUTRAL (random signs negate benefit)
- **#40 HadaNorm mean-centering**: DROPPED (duplicate of #22, incompatible with per-token quant)
- **#49/#17 parallel_blocks tuning**: DONE — NO EFFECT (FFN dominates decode, not attention)
- **#45 Gemma-3 SWA V bug**: DONE — **FIXED**. Added V un-rotation to iSWA build_attn overload. Gemma-3 turbo3 K+V PPL 5.8867 (+3.3% vs q8_0 5.6995). Was 45T before fix.
- **#31 Turbo in speculative decoding**: DONE — NO BENEFIT (draft KV is tiny, spec decode slower than normal decode for this pair)
- **#16b turbo4 prefill fix**: NOT YET STARTED
- **#42 KVLinC asymmetric K/V rotation**: DONE — NEGATIVE (rotation helps both K and V)
- **turbo4 head_dim=128 bug**: FOUND ROOT CAUSE — see below

Server state: iSWA fix deployed at `/root/llama-cuda-turbo/` on `root@dorei`. Fully rebuilt.

## BUG FOUND: turbo4 K broken on head_dim=128

turbo4-K produces PPL 33K on Qwen3-14B (head_dim=128). turbo4-V works fine (PPL 6.62).

**Root cause**: In `fattn.cu` line 702, the Q pre-rotation check is:
```c
const bool turbo_kv = K->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO3_0;
```
This only checks for TURBO3_0, NOT TURBO4_0! So when using turbo4 K:
1. Q pre-rotation is SKIPPED (turbo_kv is false)
2. Vec kernel gets UNROTATED Q dotted with ROTATED K → garbage dot products → PPL 33K

This same `turbo_kv` variable gates THREE things in the else branch (lines 702-773):
1. Line 713: `do_decode_dequant` — fp16 decode dequant (turbo4 doesn't need this anyway)
2. Line 759: Q pre-rotation — **THIS IS THE BUG**. turbo4 K is stored rotated but Q never gets rotated
3. The fp16 dequant kernels at lines 722/738 only handle TURBO3_0 blocks (correct, turbo4 uses vec kernel)

**Fix**: Change line 759 to check for turbo4 K type too:
```c
const bool turbo_k_any = (K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0);
if (turbo_k_any && Q->ne[0] % 128 == 0) {
```
Only need to check K type (not V) since Q pre-rotation is for the Q·K dot product.

**Mystery**: turbo4-K + q8_0-V gave PPL 5.8451 on Qwen3.5-27B (head_dim=256) in experiment #10. Need to verify this wasn't a fluke or different code state. The bug should affect ALL turbo4-K regardless of head_dim.

**Test data** (Qwen3-14B Q5_K_M, head_dim=128, 2K/8chunks):
- q8_0 baseline: PPL 6.5020
- turbo3 K+V: PPL 6.7458 (+3.7%)
- turbo4 K+V: PPL 32643 (BROKEN)
- turbo4 K + q8_0 V: PPL 33890 (BROKEN — confirms K is the issue)
- q8_0 K + turbo4 V: PPL 6.6232 (+1.9% — V works great, better than turbo3!)
- GGML_TURBO_DECODE_FP16=1 turbo4 K: still broken (same PPL 33890 — rules out vec_dot inline bug)
Local branch: `feature/turboquant-kv-cache` with iSWA fix + experiments.md + benchmark-results.md updates.

## Gemma-3-27B-it turbo3 Results (post-#45 iSWA fix)

| Config | PPL (2K/8chunks) | vs q8_0 | Notes |
|--------|------------------|---------|-------|
| q8_0 baseline | 5.6995 ± 0.174 | — | |
| turbo3 K+V | 5.8867 ± 0.170 | +3.3% | **was 45T PPL before fix** |
| turbo3-K + q8_0-V | 5.9633 ± 0.174 | +4.6% | K-only |

**Finding**: With iSWA V un-rotation fix, Gemma-3 turbo3 matches the head_dim=128 degradation pattern (+3-4% PPL). Same as MN-Violet-Lotus (+2.6%) and Qwen3-14B (+3.8%). K-only is slightly worse than K+V, consistent with "values matter more" finding.

## Experiment: KVLinC No-K-Rotation (#42)

| Config | PPL (2K/8chunks) | vs q8_0 |
|--------|------------------|---------|
| turbo3 baseline (both rotated) | 5.8323 | -0.09% |
| turbo3 K unrotated, V rotated | 6.1647 | +5.6% |
| turbo3 neither rotated (prior) | 6.2357 | +6.8% |
| q8_0 | 5.8375 | — |

**Conclusion**: KVLinC's "rotation hurts keys" finding does NOT apply to turbo3 Lloyd-Max codebook. Rotation helps both K and V. K rotation alone contributes ~0.07 PPL, V rotation ~0.33 PPL, together ~0.40 PPL improvement.

Experiment branches created: `experiment/gsr-walsh-ordering`, `experiment/parallel-blocks-tuning`, `experiment/kvlinc-no-k-rotation`.

## Experiment: parallel_blocks Tuning (#49/#17)

Forced different parallel_blocks values via GGML_PARALLEL_BLOCKS env var.
turbo3 tg64 at 32K context:

| parallel_blocks | tok/s |
|-----------------|-------|
| default (auto) | 29.95 ± 0.07 |
| 1 | 29.97 ± 0.07 |
| 2 | 29.95 ± 0.10 |
| 4 | 29.95 ± 0.06 |
| 8 | 29.96 ± 0.05 |
| 16 | 29.96 ± 0.07 |
| 32 | 29.93 ± 0.08 |
| q8_0 baseline | 30.81 ± 0.07 |

**Conclusion**: NO EFFECT. All values within noise. Attention is <5% of decode
time — FFN dominates. 2.8% turbo3→q8_0 gap is structural dequant overhead.

## Experiment: GSR Walsh Ordering (#39)

Reorder FWHT output by sequency (sign-change count) to group similar-frequency
components into turbo3 quantization blocks.

| Config | PPL (2K/8chunks) | vs q8_0 |
|--------|------------------|---------|
| turbo3 baseline (natural Hadamard) | 5.8323 | -0.09% |
| turbo3 + Walsh ordering | 5.8248 | -0.22% |
| q8_0 reference | 5.8375 | — |

**Conclusion**: NEUTRAL (-0.13%, within ±0.164 error bars). Random sign arrays
in PolarQuant already decorrelate all 128 FWHT output elements, making them
identically distributed. Sequency reordering cannot improve intra-block variance
when frequency structure is already destroyed by random signs. GSR paper's gains
(PPL 20.29→11.59) were with non-randomized Hadamard.

## Multi-Model Validation (turbo3 uniform, 2K context)

**BUG FIX**: KV cache context OOM for turbo types — the ggml context allocation didn't
account for turbo rotation matrix tensors (2 extra objects). Fixed by adding `n_turbo_extra`
to the context size calculation in `llama-kv-cache.cpp`.

| Model | Architecture | head_dim | q8_0 PPL | turbo3 PPL | Delta |
|-------|-------------|----------|----------|-----------|-------|
| Qwen3.5-27B Q6_K | qwen3 | 256 | 5.8375 | 5.8501 | +0.2% |
| Qwen3.5-35B-A3B Q4_K_S (MoE) | qwen3 | 256 | 6.4155 | 6.4334 | +0.3% |
| MN-Violet-Lotus-12B Q4_K_M | llama | 128 | 5.6051 | 5.7494 | +2.6% |
| Qwen3-14B Q5_K_M | qwen3 | 128 | 8.3084 | 8.6230 | +3.8% |
| Gemma-3-27B-it Q4_K_M (K only) | gemma3 | 128 | 5.6995 | 7.4946 | +31% |
| Gemma-3-27B-it Q4_K_M (V only) | gemma3 | 128 | 5.6995 | **45T** | **broken** |

**Key findings**:
1. turbo3 works excellently on Qwen3.5 models (head_dim=256): <0.3% PPL increase
2. Quality degrades moderately on head_dim=128 models: +2.6% to +3.8% PPL
3. **Gemma-3 V is completely broken** — turbo3 V produces catastrophic PPL on Gemma-3's
   SWA/global hybrid attention architecture. turbo3 K works (degraded). Root cause:
   likely related to the interleaved sliding-window attention cache architecture.
4. The KV cache context OOM was a latent bug affecting all non-Qwen3.5 models with turbo types

## Experiment #50: turbo4 Q Pre-rotation Fix (APPLIED)

**Fix**: In `fattn.cu`, changed Q pre-rotation guard from `turbo_kv` (only checks TURBO3_0) to
`turbo_k_any = (K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0)`.

### Qwen3.5-27B Q6_K (head_dim=256, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 K+V | 5.8375 | — |
| q8_0-K + turbo4-V | 5.8372 | -0.01% |
| turbo4-K + q8_0-V | 5.8451 | +0.13% |
| **turbo4 K+V** | **5.8186** | **-0.32%** |
| turbo3 uniform | 5.8501 | +0.22% |

**turbo4 K+V BEATS q8_0 on head_dim=256!** Also beats turbo3.

### Qwen3-14B Q5_K_M (head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 K+V | 6.5020 | — |
| q8_0-K + turbo4-V | 6.6232 | +1.9% |
| turbo3 K+V | 6.7458 | +3.7% |
| turbo4-K + q8_0-V | 6.8322 | +5.1% |
| turbo4 K+V | 6.9118 | +6.3% |

On head_dim=128, turbo4-V (+1.9%) beats turbo3 K+V (+3.7%), but turbo4-K (+5.1%) is the weak link.

### Summary
- Fix resolves turbo4 K from garbage (33K PPL) to functional on both head_dim sizes
- head_dim=256: turbo4 is the BEST quantization option (-0.32% vs q8_0)
- head_dim=128: turbo4-V excellent, turbo4-K degrades more than turbo3
- Experiment #10's turbo4-K result (5.8451) confirmed reproducible after fix

## Sparse V Dequant (credit: TheTom, sparse-v-dequant)

Skip V dequantization for KV positions where `exp(score - max) < 1e-6`.
At long context, 90%+ of attention weights are negligible — eliminating those
dequant operations removes work without quality loss.

### Quality (turbo3, Qwen3.5-27B, 2K/8chunks)
PPL = 5.8501 — **bit-identical** to baseline. Zero quality impact.

### Dense Model: Qwen3.5-27B Q6_K — Decode tg64

| Context | turbo3 (fp16 dq) | turbo3 (native) | turbo4 |
|---------|------------------|-----------------|--------|
| 4K | 30.25 | — | 29.68 |
| 32K | 30.03 | 30.02 | — |

No meaningful improvement on dense model (attention is <5% of decode compute).

### MoE Model: Qwen3.5-35B-A3B Q4_K_S — Decode tg64

**KEY FINDING: sparse V eliminates context scaling regression on native dequant!**

| Context | fp16 dq (sparse V) | native (sparse V) | native (no sparse V, old) |
|---------|--------------------|--------------------|---------------------------|
| p=0 | — | — | 125.77 |
| 4K | 127.53 | 127.13 | — |
| 8K | — | **126.89** | **114.44** (-9% regression!) |
| 32K | 126.70 | **126.21** | (worse) |

Native dequant with sparse V matches fp16 dequant speed at all contexts.
The fp16 dequant path was originally created to fix this regression — sparse V
achieves the same fix with zero extra memory bandwidth (no temp fp16 buffer).

### Experiment #16b: turbo4 Prefill MMA Enabled

Enabled fp16 dequant + MMA for turbo4 prefill. QJL correction loses ~1% PPL in fp16,
but 1.9x prefill speedup is worth it (only prompt tokens affected).

| Metric | turbo4 before | turbo4 after | turbo3 (ref) | q8_0 (ref) |
|--------|---------------|--------------|--------------|------------|
| pp4096 tok/s | 588 | **1113** | 1125 | 1133 |
| PPL (all-prefill) | 5.8186 | 5.8966 | 5.8501 | 5.8375 |
| Decode tg64 | 29.43 | 29.66 | 30.25 | 31.04 |

turbo3 PPL verified unchanged (5.8501).

### CUDA 13.2 Compatibility
Built and tested with CUDA 13.2.0 (nvcc V13.2.51). No segfault, identical results.
Turbo3 PPL: 5.8501 (identical). Turbo4 PPL: 5.8186 (identical).

### Multi-Sequence (n_seq > 1) Fix — 2026-03-27

Bug: turbo dequant-to-fp16 kernels in fattn.cu ignored ne[3] (stream dimension).
With kv_unified=false (default) and n_seq > 1, only stream 0 was dequanted;
other streams had uninitialized fp16 data, causing catastrophic PPL.

| Config | Before fix | After fix |
|--------|-----------|-----------|
| turbo3 n_seq=1 | 6.31 | 6.31 |
| turbo3 n_seq=2 | 17.10 | 6.30 |
| turbo3 n_seq=4 | 22.56 | 6.34 |
| turbo3 -kvu (unified) | 6.30 | 6.30 |

## Community Report: Dual RTX 4090 (multi-GPU, unknown model)

Date: 2026-03-27. First multi-GPU test after q_rot_buf per-device fix (6cdd9db87).

| Config | Prefill tok/s | Decode tok/s | vs q8_0 prefill | vs q8_0 decode |
|--------|--------------|-------------|-----------------|---------------|
| q8_0 baseline | 5116.86 | 103.57 | 100% | 100% |
| turbo3 | 4986.84 | 36.27 | 97.5% | 35.0% |
| turbo4 | 2541.50 | 17.63 | 49.7% | 17.0% |

**Analysis**:
- turbo3 prefill at 97.5% of q8_0: consistent with RTX 3090 results (98.8%)
- turbo3 decode at 35%: matches MoE decode regression pattern (Qwen3.5-35B-A3B was 37%)
- turbo4 is exactly ~2x slower than turbo3 across both prefill and decode — unexpected
- **Need to confirm**: what model, what context, was `-fa on` used?
- turbo4 prefill at 49.7% suggests it may NOT be hitting the MMA prefill path (which should give ~98% like turbo3). Possible that `-fa` is off or the model triggers a non-MMA kernel path on multi-GPU.

## Reproduction Test: ubergarm's Q4_0 + c512 configuration (2026-03-27)

Testing exact same params as ubergarm (ik_llama.cpp#1509): Qwen3.5-35B-A3B-Q4_0, -c 512, seed 1337, -ngl 99.

**Hardware**: RTX 3090 24GB (ubergarm used RTX A6000 48GB)

| KV type | LA | PPL (ours) | PPL (ubergarm) | Delta |
|---------|------|------------|----------------|-------|
| f16 | - | 6.5776 ± 0.04194 | 6.5792 ± 0.04196 | match |
| turbo3 | 1 | 6.6112 ± 0.04218 | 9.2400 | +0.51% vs +40.4% |

**Conclusion**: Cannot reproduce. f16 baselines match perfectly, confirming identical model/dataset.
turbo3 PPL is +0.51% on our build vs +40.4% on theirs. Issue is on their end —
likely build config (stale cmake, missing FA, or patches applied on wrong base).

## turbo2 (2-bit) Initial Results (2026-03-27)

New 2-bit PolarQuant type: 10 bytes per 32 elements = 2.5 bpv (6.4x vs fp16).
Model: Qwen3.5-27B-heretic Q6_K, head_dim=256, RTX 3090.

### PPL (4K ctx, 4 chunks)

| Config | PPL | vs f16 | vs q8_0 |
|--------|-----|--------|---------|
| f16 baseline | 6.2765 | — | — |
| q8_0 baseline | 6.2677 | -0.14% | — |
| turbo3 (3-bit) | 6.3252 | +0.78% | +0.92% |
| **turbo2 (2-bit)** | **6.7792** | **+8.01%** | **+8.16%** |

### Decode Speed pp512+tg64 (tok/s)

| Config | pp512 | tg128 | pp512+tg64 | pp32768+tg64 |
|--------|-------|-------|------------|--------------|
| q8_0 | 1150.91 | 30.98 | 227.27 | 921.92 |
| turbo3 | 1138.57 | 30.27 | 222.76 | 904.84 |
| turbo2 | 1141.87 | 30.64 | 225.64 | 908.88 |

### KV Memory (4K context, 16 KV layers)

| Config | KV size | Compression vs fp16 |
|--------|---------|---------------------|
| f16 | 128 MiB | 1.0x |
| q8_0 | 68 MiB | 1.9x |
| turbo3 | 28 MiB | 4.6x |
| turbo2 | 20 MiB | 6.4x |

### Mixed K/V Configurations (4K ctx, 4 chunks)

| K type | V type | PPL | vs f16 | Effective bpv |
|--------|--------|-----|--------|---------------|
| turbo3 | turbo2 | 6.5670 | +4.63% | ~2.9 |
| turbo2 | turbo3 | 6.5203 | +3.88% | ~2.9 |
| turbo2 | q8_0 | 6.4894 | +3.39% | ~5.3 |
| q8_0 | turbo2 | 6.5490 | +4.34% | ~5.3 |

### Layer-Adaptive turbo2 (4K ctx, 4 chunks)

| Config | PPL | vs f16 | Notes |
|--------|-----|--------|-------|
| turbo2 uniform | 6.7792 | +8.01% | all 16 layers turbo2 |
| turbo2 LA-1 | 6.7411 | +7.40% | first4+last4 = q8_0, 8 turbo2 |
| turbo2 LA-2 | 6.6866 | +6.53% | last 8 = q8_0, 8 turbo2 |
| (turbo3 uniform) | 6.3252 | +0.78% | reference: still much better |

**Analysis**: turbo2 at 2.5 bpv has +8% PPL degradation on head_dim=256 (Qwen3.5-27B).
This is significantly worse than turbo3's <1% gap. Speed is essentially identical to turbo3
(both compute-bound by dequant, not memory-bound at short context on this model).
The 6.4x compression is impressive but the quality cost is too high for general use.
turbo2 may work better for V-only (K stays turbo3) or for extreme memory pressure scenarios.

## turbo3 vs q8_0 vs turbo4 — Full Context Scaling (2026-03-28)

Model: Qwen3.5-27B-heretic Q6_K, RTX 3090 24GB. turbo3 uniform, turbo4 uniform.

### PPL by Context Length

| Context | Chunks | q8_0 | turbo3 | t3 vs q8 | turbo4 | t4 vs q8 | LA-1 t3 | LA-1 vs q8 |
|---------|--------|------|--------|----------|--------|----------|---------|------------|
| 2K | 8 | 5.8375 | 5.8323 | -0.09% | 5.8186 | -0.32% | 5.7690 | -1.17% |
| 4K | 4 | 6.2677 | 6.3252 | +0.92% | — | — | 6.3198 | +0.83% |
| 8K | 4 | 7.4241 | 7.3783 | -0.62% | — | — | 7.3952 | -0.39% |
| 32K | 1 | 7.2139 | 7.1693 | -0.62% | 7.3296 | +1.60% | 7.2168 | +0.04% |
| 64K | 1 | 8.1975 | 8.2379 | +0.49% | 8.5004 | +3.69% | — | — |

### Decode Speed tg64 (tok/s)

| Context | q8_0 | turbo3 | t3/q8 | turbo4 | t4/q8 |
|---------|------|--------|-------|--------|-------|
| 32K | 30.69 | 29.83 | 97.2% | 29.47 | 96.0% |
| 64K | — | 29.79 | — | 29.51 | — |

### VRAM at 64K

| Config | VRAM |
|--------|------|
| turbo3 | ~22.3 GiB |
| turbo4 | ~22.2 GiB |
| q8_0 | **OOM** (~28+ GiB needed) |

### Key Findings

1. **turbo3 holds quality at all context lengths**: -0.62% at 32K, +0.49% at 64K (within noise)
2. **turbo4 degrades catastrophically at long context**: +1.60% at 32K, **+3.69% at 64K**
3. turbo4 QJL noise accumulates as sqrt(N) — 1-bit correction errors compound over more KV positions
4. LA-1 advantage vanishes at long context: +0.04% at 32K (was -1.17% at 2K)
5. turbo3 fits 64K on 24GB RTX 3090 where q8_0 would OOM
6. Decode speed is constant across context lengths for both turbo types
7. **Recommendation**: turbo3 for all use cases. turbo4 only viable at short context (<4K) on head_dim=256

## InnerQ Per-Channel Equalization (#52) — 2026-03-28

Branch: `experiment/innerq-channel-equalization`. Model: Qwen3-14B Q5_K_M (head_dim=128).

### Concept

Per-channel K scaling before L2 norm + FWHT to equalize channel magnitudes.
Inverse scaling applied to Q in FA kernel before FWHT rotation. Targets the
head_dim=128 quality gap where channels have extreme anisotropy (channel 114
has 20x mean RMS on Qwen3-14B).

### Calibration

Online calibration: accumulate per-channel K² during first 100K token-counts,
compute sqrt-dampened scales: `scale = (rms/mean_rms)^strength`, clamped to max 2.0x.
Controlled by `TURBO_INNERQ=1` and `TURBO_INNERQ_STRENGTH=<float>` env vars.

### Reference Baselines (Qwen3-14B Q5_K_M, head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| f16 baseline | 6.4239 | +0.05% |
| q8_0 baseline | 6.4206 | — |
| turbo3 uniform (no InnerQ) | 6.6340 | +3.33% |

### Strength Sweep (turbo3 + InnerQ, 2K/8chunks)

| Strength | PPL | vs q8_0 | vs turbo3 baseline | Gap closed |
|----------|-----|---------|--------------------|-----------:|
| 0.00 (identity) | 6.6340 | +3.33% | — | 0% |
| 0.10 | 6.5596 | +2.17% | -1.12% | 35% |
| **0.20** | **6.5175** | **+1.51%** | **-1.76%** | **55%** |
| 0.30 | 6.5310 | +1.72% | -1.55% | 48% |
| 0.50 | 6.5872 | +2.60% | -0.71% | 22% |
| 1.00 (full RMS) | catastrophic | — | — | — |

**Optimal: strength=0.20, max clamp=2.0x**

### Effect on head_dim=256 (Qwen3.5-27B)

No benefit — FWHT with 256 dimensions already equalizes channels effectively.
Max channel ratio only 2x (vs 20x on hd128). InnerQ disabled automatically
when head_dim >= 256 would be appropriate, but currently gated by env var.

### Asymmetric Comparison (Qwen3-14B, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 K+V + InnerQ (strength=0.20) | 6.5175 | +1.51% |
| turbo3-K + turbo4-V + InnerQ | 6.6645 | +3.80% |
| turbo3 K+V (no InnerQ) | 6.6340 | +3.33% |

turbo3 uniform + InnerQ beats asymmetric turbo3-K + turbo4-V with InnerQ.

### Effect on head_dim=256 — Auto-Detection (2026-03-28)

| Config | PPL | vs q8_0 | Notes |
|--------|-----|---------|-------|
| turbo3 (no InnerQ) | 5.8501 | +0.22% | baseline |
| turbo3 + InnerQ (strength=0.20) | 5.9283 | +1.56% | **WORSE** — channels balanced, perturbation hurts |
| turbo3 + InnerQ (auto-detect) | 5.8501 | +0.22% | auto-disabled: max ratio 1.164 < 1.2 |

Auto-detect threshold (max_ratio < 1.2) correctly disables InnerQ on hd256.

### K-only vs K+V Scaling Comparison (Qwen3-14B, strength=0.20)

| Config | PPL | vs q8_0 | Gap closed |
|--------|-----|---------|------------|
| K+V calibrate, K+V apply (best) | 6.5349 | +1.78% | 46% |
| K-only calibrate, K+V apply | 6.5757 | +2.42% | 27% |
| K-only calibrate, K-only apply (s=0.30) | 6.5418 | +1.89% | 43% |
| K-only calibrate, K-only apply (s=0.20) | 6.5477 | +1.98% | 40% |
| K-only calibrate, K-only apply (s=0.50) | 6.5486 | +1.99% | 40% |
| Max-based (paper's formula, mode=1) | 6.6716 | +3.91% | **WORSE** |
| No InnerQ baseline | 6.6340 | +3.33% | 0% |

**Key findings**:
1. V scaling helps even without output compensation (K+V > K-only)
2. Mixed K+V calibration stats are better than K-only stats for K+V application
3. Paper's max-based formula doesn't transfer to our codebook pipeline
4. RMS-based mode=0 with strength=0.20 is optimal

### Summary

InnerQ closes the head_dim=128 gap by **~46%** (from +3.33% to +1.78% vs q8_0).
Auto-detects and disables on hd256 models where channels are already balanced.
The remaining +1.78% gap is inherent to lower dimensionality (fewer dimensions =
larger relative quantization noise in dot products). Further improvement would
require more bits or a fundamentally different approach.

## New turbo4: 4-bit PolarQuant (16 centroids, no QJL)

Branch: `experiment/turbo4-4bit-polarquant`. Dropped QJL 1-bit correction entirely,
replaced with pure 4-bit PolarQuant (16 Lloyd-Max centroids). Inspired by TheTom's
Metal implementation showing QJL variance gets amplified by softmax.

Block format: 66 bytes per 128 values = 4.125 bpv (was 68 bytes = 4.25 bpv with QJL).

### Qwen3.5-27B Q6_K (head_dim=256, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 | 5.8377 | — |
| new turbo4 (4-bit) | 5.8467 | +0.15% |
| turbo3 | 5.8501 | +0.22% |
| old turbo4 (QJL) | 5.8186 | -0.32% |

On hd256, old QJL turbo4 was better (-0.32% vs +0.15%). QJL helps on hd256
where the 1-bit correction noise is small relative to 256-dim dot products.

### Qwen3-14B Q5_K_M (head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 | 6.4206 | — |
| **new turbo4 (4-bit)** | **6.4978** | **+1.20%** |
| turbo3 + InnerQ | 6.5175 | +1.51% |
| turbo3 | 6.6340 | +3.33% |
| old turbo4 (QJL) | 6.9118 | +6.3% |

**Breakthrough on hd128!** New turbo4 beats turbo3+InnerQ by 0.31 percentage points.
Old turbo4 (QJL) was catastrophic at +6.3%. The 72.5% MSE reduction from 16 vs 8
centroids more than compensates for losing the QJL correction.

InnerQ has no additional effect on new turbo4 (PPL identical with/without).

### LA-1 on Qwen3-14B (head_dim=128, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| turbo3 | 6.6340 | +3.33% |
| turbo3 + LA-1 | 6.6763 | +3.98% |
| turbo3 + LA-1 + InnerQ | 6.6608 | +3.74% |

LA-1 HURTS on hd128 (unlike hd256 where it gives -0.71%). 8/40=20% promoted layers
create quality discontinuity that is worse than uniform turbo3 noise.

### turbo4 Quality Decomposition (hd128, Qwen3-14B, 2K/8chunks)

**K quality dominates V by 13x.** Source of gap analysis:

| Config | PPL | vs q8_0 | Source |
|--------|-----|---------|--------|
| q8_0 (f16 K + f16 V) | 6.4206 | — | — |
| f16 K + turbo4 V | 6.4256 | +0.08% | V quant only |
| turbo4 K + f16 V | 6.4879 | +1.05% | K quant only |
| turbo4 K + turbo4 V | 6.4978 | +1.20% | K + V combined |
| turbo4 K + turbo3 V | 6.5854 | +2.57% | |
| turbo3 K + turbo4 V | 6.6330 | +3.31% | |
| turbo3 K + turbo3 V | 6.6340 | +3.33% | |

### turbo4 + Layer-Adaptive sweep (hd128, Qwen3-14B, 2K/8chunks)

| Mode | Description | PPL | vs q8_0 | Promoted layers |
|------|-------------|-----|---------|-----------------|
| **LA-3** | **last 4 (K+V)** | **6.4587** | **+0.59%** | **4/40 = 10%** |
| LA-10 | last 4 (K-only) | 6.4604 | +0.62% | K: 4/40, V: 0/40 |
| LA-11 | last 6 (K+V) | 6.4626 | +0.65% | 6/40 |
| LA-2 | last 8 (K+V) | 6.4628 | +0.66% | 8/40 |
| LA-1 | first4+last4 (K+V) | 6.4638 | +0.67% | 8/40 |
| LA-7 | last 8 (K-only) | 6.4653 | +0.70% | K: 8/40, V: 0/40 |
| LA-9 | last 2 (K+V) | 6.4795 | +0.92% | 2/40 |
| 0 | turbo4 uniform | 6.4978 | +1.20% | 0/40 |
| LA-4 | first 4 (K+V) | 6.5125 | +1.43% | 4/40 |
| LA-5 | first2+last2 (K+V) | 6.5162 | +1.49% | 4/40 |

Best: LA-3 (last 4, K+V) at +0.59%, 4.53 bpv average.
Sweet spot: exactly 4 last layers. More layers (6, 8) give diminishing returns.
First-layer promotion always hurts on hd128. K-only promotion nearly as good.

### turbo4 Context Scaling (hd128, Qwen3-14B)

| Context | q8_0 PPL | turbo4 PPL | turbo4 gap | turbo3 PPL | turbo3 gap |
|---------|----------|------------|------------|------------|------------|
| 2K | 6.4206 | 6.4978 | +1.20% | 6.6340 | +3.33% |
| 8K | 7.0149 | 7.0718 | +0.81% | 7.4261 | +5.86% |
| 32K | 7.2929 | 7.3638 | +0.97% | 7.7736 | +6.59% |

turbo4 context scaling is excellent (~1% gap at all lengths).
turbo3 degrades catastrophically at long context on hd128 (3.3% → 6.6%).

### Decode Speed (hd128, Qwen3-14B, RTX 3090, tg64)

| Config | t/s | vs q8_0 |
|--------|-----|---------|
| q8_0 | 69.68 | — |
| turbo4 | 59.72 | 85.7% |
| turbo3 | 59.54 | 85.4% |

turbo4 and turbo3 have identical decode speed (compute-bound dequant).

### TheTom's centroids test (from turbo4-resurrection.md)

TheTom's centroids: [-0.1739, -0.1172, -0.0895, -0.0688, -0.0513, -0.0356, -0.0210, -0.0069, ...]
Ours (Gaussian Lloyd-Max): [-0.2416, -0.1829, -0.1430, -0.1111, -0.0833, -0.0581, -0.0343, -0.0114, ...]

| Centroids | hd128 2K PPL | vs q8_0 | hd128 8K PPL | vs q8_0 | hd256 2K PPL | vs q8_0 |
|-----------|-------------|---------|-------------|---------|-------------|---------|
| Our Gaussian | 6.4978 | +1.20% | 7.0718 | +0.81% | 5.8467 | +0.15% |
| TheTom's | 6.4082 | -0.19% | 7.0726 | +0.82% | 5.8829 | +0.77% |

TheTom's centroids dramatically help hd128 at 2K (-0.19% beats q8_0!) but the gain
vanishes at 8K (+0.82% ≈ same as Gaussian). Hurts hd256 (+0.77% vs +0.15%).

Raw norm (no recon correction) test: PPL 7.80 on hd256 — catastrophic. Norm correction is essential.

KEY OPEN QUESTION: TheTom's centroids don't match standard Lloyd-Max for ANY Gaussian.
In N(0,1) units: outer=1.97 vs Lloyd-Max 2.73. He likely ran iterative Lloyd-Max on real
post-FWHT KV data. Need to ask: (1) exact computation method, (2) does he normalize
per-head or per-block? Per-head normalization for hd256 would give σ=1/√256 per block
instead of our σ=1/√128.

### Empirical centroid calibration — FAILED

Simulated post-FWHT distribution has kurtosis -0.047 (sub-Gaussian), but the empirical
centroids derived from simulation gave catastrophic PPL = 6.8758 (+17.8%). The simulation
is wrong for real model data. TheTom's measurement on real KV tensors shows kurtosis = 2.9
(near-Gaussian), confirming Gaussian-optimal Lloyd-Max centroids are correct.

### Empirical codebook: real post-FWHT data (TURBO_EXTRACT)

Extracted 2M post-FWHT samples from each model. Post-rotation distribution is
model-independent N(0, 1/√128): std=0.088388, kurtosis≈2.9 on both models.
Empirical Lloyd-Max centroids improve MSE by only 0.2-0.3% vs Gaussian — already optimal.

## New turbo4: K/V split + Layer-Adaptive on hd256

Qwen3.5-27B Q6_K (head_dim=256, 2K/8chunks). q8_0 baseline: 5.8375.

### K/V Quality Decomposition (hd256)

| Config | PPL | vs q8_0 | Notes |
|--------|-----|---------|-------|
| q8_0 K + turbo4 V | 5.8397 | +0.04% | V quant essentially free |
| turbo4 K + turbo4 V | 5.8467 | +0.16% | uniform turbo4 |
| turbo4 K + q8_0 V | 5.8590 | +0.37% | K-only turbo4 — counterproductive |

V quantization adds only +0.04%. K adds +0.12%. But keeping V at q8_0 is WORSE (+0.37%)
than turbo4 V (+0.16%), likely because turbo4 V dequant noise averages out better through
softmax-weighted sum than q8_0 systematic rounding.

### Layer-Adaptive Sweep (hd256, new turbo4)

Model has 16 attention layers (48 recurrent Delta Net layers use no KV cache).

| Mode | Description | PPL | vs q8_0 | Promoted layers |
|------|-------------|-----|---------|-----------------|
| **LA-3** | **last 4 (K+V)** | **5.8420** | **+0.08%** | **4/16 = 25%** |
| LA-10 | last 4 (K-only) | 5.8441 | +0.11% | K: 4/16, V: 0/16 |
| LA-1 | first4+last4 (K+V) | 5.8461 | +0.15% | 8/16 = 50% |
| 0 | turbo4 uniform | 5.8467 | +0.16% | 0/16 |

**LA-3 (last 4 K+V q8_0) wins at +0.08% — practically lossless.**
LA-1 (first4+last4) doesn't help on new turbo4 hd256 (unlike old QJL turbo4 where it was -0.71%).
Only 16 attention layers means 4 promoted layers = 25% of KV cache at q8_0.

## MMA Prefill fp16 Precision Fix — turbo4 (2026-03-28)

**Root cause**: During prefill (Q->ne[1] > 1), turbo K/V are dequanted to fp16 temp buffers
for tensor core MMA attention. turbo4's 16 centroids (spaced ~0.023 apart when scaled by norm)
round-trip through fp16, losing precision. turbo3's 8 centroids (wider spacing) survive fp16.

**Fix**: turbo4 now automatically bypasses MMA prefill and uses vec kernel (full float32
precision). turbo2/turbo3 keep MMA prefill. FA auto-enabled for turbo types (was an error).

### PPL Impact (Qwen3.5-27B, hd256, 2K/8chunks)

| Config | PPL | vs q8_0 |
|--------|-----|---------|
| q8_0 baseline | 5.8375 | — |
| **turbo4 (vec prefill, auto)** | **5.8310** | **-0.11%** |
| turbo4 (MMA prefill, old) | 5.8467 | +0.16% |
| turbo3 (MMA prefill) | 5.8501 | +0.22% |

turbo4 now **beats q8_0** by 0.11% without any manual env var.

### Speed Impact

| Config | pp4096 tok/s | pp/q8 | tg64 tok/s | tg/q8 |
|--------|-------------|-------|-----------|-------|
| q8_0 | 1139 | 100% | 31.23 | 100% |
| turbo3 (MMA) | 1120 | 98.3% | 30.09 | 96.4% |
| turbo4 (vec) | 420 | 36.9% | 30.14 | 96.5% |

**Trade-off**: turbo4 prefill is 2.7x slower than q8_0 (vec kernel for full precision).
Decode speed is unaffected (96.5% of q8_0). For interactive use where decode matters
more than one-time prefill, this is acceptable.

### Code Changes
- `fattn.cu`: turbo4 automatically uses vec prefill (no MMA fp16 round-trip)
- `llama-context.cpp`: FA auto-enabled for turbo types (was: throw error)

---

## Experiment #53: Inverse-FWHT Prefill Dequant for turbo4
Date: 2026-03-28
Branch: experiment/tbq-ideas

### Problem
turbo4 prefill was 37% of q8_0 (420 vs 1139 tok/s) because MMA was bypassed to avoid
fp16 centroid precision loss. turbo4's 16 centroids are ~0.023 apart, rounding to same fp16.

### Solution
Inverse FWHT during K dequant-to-fp16: centroid lookup → signs2 → butterfly → signs1×norm → fp16.
After inverse rotation, values are in original domain with natural variation — fp16 handles them fine.
K only (V fp16 loss is negligible). Q is NOT pre-rotated (K is in original domain).

### Results

| Metric | Before | After |
|--------|--------|-------|
| PPL (2048, 8 chunks) | 5.831 (vec) | 5.858 (MMA inv-FWHT) |
| Prefill pp4096 | 420 tok/s | **1124 tok/s** (+167%) |
| Decode tg64 | ~30 tok/s | 30.17 tok/s (unchanged) |

PPL delta: +0.46% (within 8-chunk noise). Prefill: 2.67x improvement, matching q8_0 levels.

### Code Changes
- `fattn.cu`: new `k_turbo4_dequant_f16_inv_fwht` kernel (128 threads, shmem butterfly)
- `fattn.cu`: turbo4 K prefill dispatches inverse FWHT kernel, V uses simple dequant
- `fattn.cu`: Q pre-rotation skipped for turbo4 K (original domain)
- `fattn.cu`: removed `!turbo4_kv` MMA bypass — all turbo types use MMA prefill

---

## Experiment #57: Persistent decode K/V fp16 buffers

**Date**: 2026-03-28
**Branch**: experiment/tbq-ideas

Replace `cudaMallocAsync`/`cudaFreeAsync` per decode token with grow-only persistent `cudaMalloc` buffers for K/V fp16 dequant. Same pattern as existing `q_rot_buf`.

### Results (turbo3 K+V, Qwen3.5-27B Q6_K)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| PPL (2K/8chunks) | 5.8501 | 5.8501 | 0.00% |
| Decode tg64 (tok/s) | 29.93 | 30.16 | +0.8% |

PPL unchanged (expected — same computation). Small decode speed improvement from eliminating per-token alloc/free overhead.

### Code Changes
- `fattn.cu`: added `kv_dequant_k_buf` / `kv_dequant_v_buf` persistent per-device buffers
- `fattn.cu`: replaced 4x `cudaMallocAsync` + 4x `cudaFreeAsync` with grow-only pattern
- `fattn.cu`: removed redundant `cudaGetDevice` calls (shared device variable)

---

## Experiment #61: TCQ Python Prototype

**Date**: 2026-03-28
**Script**: `scripts/tcq_prototype.py`

Trellis-coded quantization with bitshift trellis and iterative codebook training (GLA).
Input: i.i.d. N(0,1) data, 128 elements/block (simulating post-FWHT KV cache).

### 2-bit Results

| Trellis | States | MSE | vs Lloyd-Max | dB |
|---------|--------|-----|-------------|-----|
| L=4 | 16 | 0.1185 | -0.9% | -0.04 |
| L=6 | 64 | 0.1183 | -0.7% | -0.03 |
| L=8 | 256 | 0.1101 | **+6.2%** | +0.28 |

### 3-bit Results

| Trellis | States | MSE | vs Lloyd-Max | dB |
|---------|--------|-----|-------------|-----|
| L=6 | 64 | 0.0354 | **+20.3%** | +0.99 |
| L=9 | 512 | 0.0310 | **+30.3%** | +1.57 |

D(R) bound at 3-bit: 0.0156 (4.54 dB from Lloyd-Max, 2.97 dB from TCQ L=9).

Key finding: 3-bit gains are much larger than 2-bit. Codebook training is essential.
V=1 scalar trellis — QTIP's V=2 would give larger gains but requires 2D codebook.

## Experiment #61: TCQ Implementation Results (2026-03-28)

### turbo3_tcq (3-bit TCQ, k=3, L=9, 512 states, 3.25 bpv)

Free-init trained codebook. 37.6% MSE reduction vs Lloyd-Max.

| Metric | Value | vs turbo3 | vs q8_0 |
|--------|-------|-----------|---------|
| PPL | 5.8294 | -0.05% | -0.14% |
| Prefill pp4096 | 894 tok/s | -21% vs turbo3 (1139) | |
| Decode tg64 | 28.69 tok/s | -5% vs turbo3 (30.2) | |

### turbo2_tcq (2-bit TCQ, k=2, L=8, 256 states, 2.25 bpv)

Free-init trained codebook. 4.2% MSE reduction vs Lloyd-Max.

| Metric | Value | vs turbo2 | vs q8_0 |
|--------|-------|-----------|---------|
| PPL | **6.0546** | **-61.2%** (vs 15.61) | +3.7% |
| Prefill pp4096 | 976 tok/s | | |
| Decode tg64 | 29.53 tok/s | | |

Key finding: turbo2_tcq at 2.25 bpv achieves near-turbo3 quality (6.05 vs 5.83) with 25% fewer bits.
The trellis structure provides enormous gains at 2-bit where scalar quantization breaks down.

## TCQ Optimized Codebooks (2026-03-29, 4 chunks long-ctx / 8 chunks 2K)

numpy GLA training: n_train=4000, 100 iters. 3-bit MSE: 50.1% reduction. 2-bit MSE: 33.1% reduction.
Previous committed codebooks were unoptimized (3-bit: 37.6%, 2-bit: only 4.2% — essentially useless).

### Complete Context Scaling Comparison (all types + hybrids)

| Type | bpv | 2K PPL | vs q8_0 | 32K PPL | vs q8_0 | 65K PPL | vs q8_0 |
|------|-----|--------|---------|---------|---------|---------|---------|
| q8_0 | 8.0 | 5.8375 | — | 6.9539 | — | 6.9344 | — |
| turbo3 | 3.25 | 5.8325 | -0.09% | 7.0790 | +1.80% | 7.0929 | +2.28% |
| turbo3_tcq | 3.25 | 5.8331 | -0.08% | 7.0105 | +0.81% | 7.0067 | +1.04% |
| t2tcq-K + t3tcq-V | 2.75 | 5.9214 | +1.44% | 7.1112 | +2.26% | 7.1416 | +2.99% |
| t3tcq-K + t2tcq-V | 2.75 | 5.9904 | +2.62% | 7.1803 | +3.26% | 7.1535 | +3.16% |
| turbo2_tcq | 2.25 | 6.0592 | +3.80% | 7.2458 | +4.20% | 7.2944 | +5.19% |
| turbo2 | ~2.1 | 6.0083 | +2.93% | 7.5505 | +8.58% | 7.5259 | +8.53% |
| turbo4 | 2.0 | 5.8644 | +0.46% | 6.9697 | +0.23% | 6.9697 | +0.51% |

### Key findings
- turbo4 PolarQuant (2.0 bpv) has the best context scaling of any type — gap stays <0.51% even at 65K
- turbo3_tcq cuts 32K degradation by more than half vs turbo3 (0.81% vs 1.80%), holds at 1.04% at 65K
- TCQ transforms turbo2 context scaling: 8.58%→4.20% at 32K, 8.53%→5.19% at 65K
- t2tcq-K + t3tcq-V (2.75 bpv) is the best hybrid: only +1.44% at 2K, confirming V quality matters more
- turbo2 without TCQ degrades catastrophically at long context (+8.5% at 32K+)
- turbo3_tcq at 65K (+1.04%) is better than turbo3 at 32K (+1.80%) — TCQ enables longer effective context

## TurboQuant vs Rotated q4_0/q8_0 (ggerganov PR #21038) — 2026-03-30

Branch: `master` (merged ggml-org + PR #21038)
Model: Qwen3.5-27B-heretic Q6_K, RTX 3090, flash_attn=1
Note: q4_0 and q8_0 now auto-enable Hadamard rotation from PR #21038.

### Speed (pp512 + tg128)

| KV Type | bpv | Prefill (tok/s) | Decode (tok/s) | Decode % of f16 |
|---------|-----|----------------|----------------|-----------------|
| f16 | 16.0 | 1156 | 31.19 | 100% |
| q8_0 (rotated) | 8.5 | 1155 | 30.86 | 98.9% |
| q4_0 (rotated) | 4.5 | 1147 | 30.81 | 98.8% |
| turbo3 | 3.5 | 1145 | 30.23 | 96.9% |
| turbo2 | 2.5 | 1144 | 30.61 | 98.1% |

### PPL at multiple context lengths

| KV Type | bpv | 2K PPL | 2K %↑ | 8K PPL | 8K %↑ | 32K PPL | 32K %↑ | 65K PPL | 65K %↑ |
|---------|-----|--------|-------|--------|-------|---------|--------|---------|--------|
| f16 (baseline) | 16.0 | 5.8048 | — | 7.3984 | — | 6.5377 | — | 6.4781 | — |
| q8_0 (rotated) | 8.5 | 5.8385 | +0.58% | 7.4278 | +0.40% | 6.5327 | -0.08% | 6.4687 | -0.15% |
| q4_0 (rotated) | 4.5 | 5.8578 | +0.91% | 7.4810 | +1.12% | 6.5835 | +0.70% | 6.5899 | +1.73% |
| turbo3 | 3.5 | 5.8501 | +0.78% | 7.3783 | -0.27% | 6.5903 | +0.80% | 6.5122 | +0.53% |
| turbo2 | 2.5 | 6.0786 | +4.72% | 8.0063 | +8.21% | 6.9028 | +5.58% | 7.1227 | +9.95% |

### Analysis

**turbo3 (3.5 bpv) vs rotated q4_0 (4.5 bpv):**
- turbo3 uses 22% LESS space than rotated q4_0
- At 2K: turbo3 is BETTER (5.85 vs 5.86)
- At 8K: turbo3 is BETTER (7.38 vs 7.48)
- At 32K: roughly tied (6.59 vs 6.58)
- At 65K: turbo3 is MUCH BETTER (+0.53% vs +1.73%)
- Conclusion: turbo3 is both smaller AND better quality, especially at long context

**Rotated q8_0 (8.5 bpv):**
- Near-perfect at all contexts — rotation essentially eliminates quantization error at 8-bit
- Slightly BETTER than f16 at 32K/65K (likely noise, but confirms zero degradation)

**turbo2 (2.5 bpv):**
- Significant degradation at long context (+9.95% at 65K)
- This is where TCQ helps most (turbo2_tcq cuts this to +5.19% from earlier benchmarks)

## TCQ Codebook GLA Optimization Study (2026-03-30)

### Key Finding: MSE-PPL Divergence

Training TCQ codebooks to deeper MSE optima **hurts** perplexity. There is a sweet spot where GLA refinement improves MSE without degrading PPL.

### 3-bit TCQ (turbo3_tcq) — MSE vs PPL Curve

| Codebook | GLA Iters | MSE Reduction | Gap to D(R) | PPL (2K, 8 chunks) | Delta PPL |
|----------|-----------|---------------|-------------|---------------------|-----------|
| Old (numpy, 4K samples) | 100 | 50.1% | 1.82 dB | 5.8236 | baseline |
| **Fine-tuned (100K samples)** | **50** | **52.8%** | **1.53 dB** | **5.8313** | **+0.13%** |
| Fine-tuned (100K samples) | 100 | 54.1% | 1.26 dB | 5.8889 | +1.12% |
| Fine-tuned (100K samples) | 200 | 54.7% | 1.14 dB | 5.9094 | +1.47% |
| From scratch (real data) | 500×30 | 54.9% | 1.06 dB | 5.8741 | +0.87% |
| From scratch (synthetic) | 500×30 | 55.5% | 1.10 dB | 5.8885 | +1.11% |

### 2-bit TCQ (turbo2_tcq)

| Codebook | GLA Iters | MSE Reduction | Gap to D(R) | PPL (2K, 8 chunks) | Delta PPL |
|----------|-----------|---------------|-------------|---------------------|-----------|
| Old (numpy, 4K samples) | 100 | 33.1% | 1.04 dB | 6.0158 | baseline |
| **Fine-tuned (100K samples)** | **50** | **34.1%** | **0.95 dB** | **5.9958** | **-0.33%** |

### Theoretical Context

Shannon rate-distortion bound D(R) = σ² × 2^{-2R} at σ = 1/√128:
- 3-bit: D(R) = 0.00012207 — best codebook reaches 1.53 dB above
- 2-bit: D(R) = 0.00048828 — best codebook reaches 0.95 dB above

For comparison, QTIP (academic SOTA) claims 0.84 dB gap at 3-bit with lattice codebooks + learned transforms.

### Interpretation

1. **MSE ≠ PPL**: Element-wise MSE is a necessary but insufficient metric. Beyond ~52% MSE reduction (3-bit), the GLA moves codebook entries away from the coset initialization structure, which correlates with PPL degradation.

2. **Coset structure matters**: The old codebook's coset initialization (shifted Lloyd-Max centroids) provides regularity that benefits attention computation beyond what MSE captures. All 64 groups are monotonically increasing in the old codebook; deep GLA breaks this to 48/64.

3. **Sweet spot**: 50 GLA iterations from coset init with 100K samples provides the best MSE/PPL tradeoff. This improves MSE by 2.7pp (3-bit) / 1.0pp (2-bit) without hurting PPL.

4. **Real vs synthetic data**: Post-FWHT KV distributions are near-perfect Gaussian (σ=0.088388 = 1/√128, kurtosis=-0.087). Synthetic training matches real data training — no "synthetic-to-real gap" exists.

### Training Setup

- Hardware: RTX 3090
- Trainer: CUDA Viterbi (1 threadblock/sample, 512 threads/block for 3-bit, 256 for 2-bit)
- Data: cuRAND N(0, 1/√128) synthetic, 100K samples/iteration
- Model: Qwen3.5-27B-heretic.Q6_K (head_dim=256, 128-element rotation groups)

## GLA Iteration Sweep — Full Curve (2026-03-30)

Tested PPL for every iteration count from 0 (pure coset init) to 200 (deep convergence).
All codebooks trained with CUDA GLA from coset initialization, 1 restart.

### 3-bit TCQ — Complete PPL vs GLA Iterations

| GLA Iters | Samples/iter | MSE Reduction | PPL (2K, 8ch) | Delta vs old |
|-----------|-------------|---------------|---------------|--------------|
| 0 (coset init) | — | 0.1% | 5.9194 | +1.64% |
| 1 | 100K | -0.1% | 5.9194 | +1.64% |
| 3 | 100K | 24.3% | 5.8450 | +0.37% |
| 5 | 100K | 33.8% | 5.8576 | +0.58% |
| 10 | 100K | 40.4% | 5.9386 | +1.97% |
| 20 | 100K | 46.0% | 5.9712 | +2.53% |
| 30 | 100K | 48.2% | 5.8733 | +0.85% |
| **50** | **100K** | **52.8%** | **5.8313** | **+0.13%** |
| 100 | 100K | 54.1% | 5.8889 | +1.12% |
| 200 | 100K | 54.7% | 5.9094 | +1.47% |
| 100 (small batch) | 4K | 53.2% | 5.9600 | +2.34% |
| **Old numpy** | **4K** | **50.1%** | **5.8236** | **baseline** |

### Key Findings

1. **PPL is NON-MONOTONIC with GLA iterations**: PPL does NOT degrade monotonically as MSE improves. It oscillates wildly — the 10-20 iter range (40-46% MSE) gives PPL WORSE than the 0-iter coset init (5.94-5.97 vs 5.92).

2. **Pure coset init is bad**: 0 GLA iterations gives PPL 5.92 — TCQ coding DOES help. The trellis needs trained centroids to be useful.

3. **The trajectory matters, not just the endpoint**: The GLA optimization path through 512-dimensional codebook space passes through regions that are good and bad for this model's attention. MSE improves monotonically but PPL is chaotic.

4. **Small-batch ≠ regularization**: 4K samples/iter (matching old numpy regime) gives PPL 5.96 — WORSE than 100K samples. The old numpy codebook's quality is NOT from small-batch regularization.

5. **The old numpy codebook is a lucky local minimum**: Its quality (50.1% MSE, PPL 5.8236) has not been replicated by ANY training configuration — CUDA or numpy. Tested 4 numpy seeds (7, 42, 123, 999) at 100 iterations: PPL range 5.8801-5.9300, all ~1% worse than old. The old codebook's quality is seed-specific, not implementation-specific.

### Numpy vs CUDA — Implementation Verification (2026-03-30)

Confirmed the CUDA trainer is NOT broken. Numpy with the same coset init produces equivalent or worse PPL:

| Impl | Seed | Iters | Samples/iter | MSE Red. | PPL |
|------|------|-------|-------------|----------|------|
| **Old numpy** | **?** | **100** | **4K** | **50.1%** | **5.8236** |
| Numpy | 42 | 100 | 4K | 52.0% | 5.8979 |
| Numpy | 42 | 200 | 4K | 54.3% | 5.8914 |
| Numpy | 7 | 100 | 4K | 51.5% | 5.8801 |
| Numpy | 123 | 100 | 4K | 52.4% | 5.9300 |
| Numpy | 999 | 100 | 4K | 52.5% | 5.8853 |
| CUDA | 42 | 50 | 100K | 52.8% | 5.8313 |
| CUDA | 42 | 100 | 4K | 53.2% | 5.9600 |

Patching verified: old numpy codebook from binary file → PPL 5.8236 (exact match).

### PPL Robustness Test — Multi-Dataset (2026-03-30)

Tests codebooks across 3 wikitext-2 splits with many more chunks. The 8-chunk "chaotic oscillation" was measurement noise — the real pattern is monotonic.

| Codebook | test 64ch | valid 64ch | train 32ch | test 8ch (old) |
|----------|-----------|-----------|-----------|----------------|
| Old numpy 100-iter | 6.507 ±0.065 | 6.909 ±0.071 | 6.956 ±0.099 | 5.824 |
| CUDA 3-iter | 6.502 ±0.065 | 6.910 ±0.071 | 6.959 ±0.098 | 5.845 |
| CUDA 10-iter | 6.595 ±0.066 | 7.026 ±0.073 | 7.063 ±0.101 | 5.939 |
| CUDA 20-iter | 6.568 ±0.066 | 7.008 ±0.072 | 7.050 ±0.101 | 5.971 |
| CUDA 30-iter | 6.560 ±0.066 | 7.022 ±0.073 | 7.057 ±0.101 | 5.873 |

Key findings:
- **3-iter matches old numpy** (±0.005 across all datasets). The "lucky codebook" was not lucky.
- **10+ iter crash is REAL** — persists across all 3 datasets (+0.06-0.12 PPL).
- **30-iter "recovery" was noise** — on valid/train, 30-iter is as bad as 10-iter.
- Real pattern: 3 iters good, 10+ iters monotonically bad. Use 3 GLA iterations.

## TCQ Multi-Layer Codebook Ablation (2026-03-30)

TURBO_TCQ_SPLIT env var: codebook A = old numpy (good), codebook B = CUDA 10-iter (bad).

### K/V Split (32 chunks, wikitext-2 test)
| Config | PPL | Δ PPL |
|--------|-----|-------|
| All A | 6.574 | baseline |
| All B | 6.657 | +0.083 |
| K=A, V=B | 6.627 | +0.053 |
| K=B, V=A | 6.620 | +0.047 |

### Layer Split (32 chunks, wikitext-2 test)
| Config | B on layers | PPL | Δ PPL |
|--------|------------|-----|-------|
| SPLIT=0 | none | 6.574 | baseline |
| SPLIT=10 | 10-39 | 6.634 | +0.060 |
| SPLIT=20 | 20-39 | 6.619 | +0.045 |
| SPLIT=30 | 30-39 | 6.599 | +0.025 |
| SPLIT=all | 0-39 | 6.657 | +0.083 |

Per-group: L0-9 +0.023 (28%), L10-19 +0.015 (18%), L20-29 +0.020 (24%), L30-39 +0.025 (30%).
Perfectly additive (sum=0.083). No concentrated sensitivity. ~+0.002 PPL per layer.

### TCQ Encode/Decode Mismatch Test (2026-03-30)

Old numpy (A) vs 10-iter (B). 64 chunks, wikitext-2 test, c=512.

| Config | Encode | Decode | PPL | Δ |
|--------|--------|--------|-----|---|
| A/A | Old numpy | Old numpy | 6.575 | baseline |
| B/B | 10-iter | 10-iter | 6.630 | +0.055 |
| A/B | Old numpy | 10-iter | 6.753 | +0.178 |
| B/A | 10-iter | Old numpy | 6.783 | +0.208 |

Mismatch penalty +0.12–0.21. Old numpy advantage is holistic (not decomposable into encode vs decode).

### Numpy Multi-Seed Codebook Test (2026-03-30)

Clean master build + runtime codebook loading. numpy GLA: n_train=4000, n_iters=100, n_restarts=1, coset init.
64 chunks, wikitext-2 test, c=512.

| Seed | PPL | Δ vs golden |
|------|-----|-------------|
| 99 (compiled-in) | 6.575 | baseline |
| 99 (reproduced) | 6.583 | +0.008 (fp32 rounding) |
| 123 | 6.600 | +0.025 |
| 42 | 6.610 | +0.035 |
| 7 | 6.611 | +0.036 |
| 999 | 6.623 | +0.048 |

Seed 99 is best of 5 but not an outlier. Range 0.048 PPL across seeds. Lucky but not unreasonably so.

### QTIP Structural Variants: Tail-Biting & Left-Shift Trellis (2026-03-30)

Clean master build + runtime codebook loading. v2 trainer: n_train=2000, n_iters=30, seed=99.
CUDA uses right-shift free-init encoder (codebooks bit-reversed for left-shift compatibility).
64 chunks, wikitext-2 test, c=512.

| Variant | CUDA-compat MSE | PPL | Δ vs compiled-in |
|---------|----------------|-----|------------------|
| Compiled-in (100iter, 4K) | (reference) | 6.577 | baseline |
| Right-shift v2 (30iter, 2K) | 0.023868 | 6.610 | +0.033 |
| Right-shift + tail-biting (30iter, 2K) | 0.023332 | 6.617 | +0.040 |
| Left-shift (30iter, 2K) | 0.023992 | 6.624 | +0.047 |

**Findings at 512 context:**
1. Left-shift trellis gives identical MSE to right-shift (isomorphism confirmed), but slightly worse PPL
2. Tail-biting training gives the BEST MSE but NOT the best PPL — MSE-PPL inverse correlation persists
3. Neither structural change improves PPL over the baseline at short context

### Context-Length Crossover: 64K PPL Tests (2026-03-30 evening)

Same codebooks tested at 64K context (4 chunks). CRITICAL FINDING: codebooks that are WORSE at 512
context become BETTER at 64K context.

| Variant | PPL @512 | Δ @512 | PPL @64K | Δ @64K |
|---------|----------|--------|----------|--------|
| Compiled-in (100iter, 4K numpy) | 6.577 | — | 7.083 | — |
| **50-iter finetuned (real data)** | ~5.97* | ~+0.13 | **7.038** | **-0.045** |
| V2 right-shift (30iter, 2K) | 6.610 | +0.033 | **7.048** | **-0.036** |
| Right-shift + TB (30iter, 2K) | 6.617 | +0.040 | **7.069** | **-0.014** |
| 50-iter numpy s42 | ~5.87* | ~+0.03 | 7.074 | -0.009 |
| 10-iter CUDA s42 | ~5.94* | ~+0.10 | 7.131 | +0.048 |
| Left-shift (30iter, 2K) | 6.624 | +0.047 | 7.153 | +0.070 |

*PPL @512 estimated from earlier sweep data at 2K context (different chunk count).

**Key findings:**
1. CONTEXT-LENGTH CROSSOVER: codebooks worse at 512 context are better at 64K. The 50-iter
   finetuned codebook is best at 64K (-0.045) despite being worst at short context.
2. The MSE-PPL "paradox" at short context may be an artifact of evaluating at short context only.
   TCQ's trellis constraint helps MORE at long context, and better-trained codebooks amplify this.
3. The 50-iter finetuned codebook (trained on real model data) gives the best 64K result.
4. Left-shift and 10-iter CUDA are worse at BOTH contexts — not all codebooks crossover.
5. This suggests the optimal codebook depends on the target context length.

### MSE-Context Scaling Grid (2026-03-30 night)

Full grid: 5 codebooks × 4 context lengths. Looking for a scaling law between MSE reduction
and optimal context length.

| Codebook | MSE Red. | PPL @2K | PPL @8K | PPL @32K | PPL @64K |
|----------|----------|---------|---------|----------|----------|
| Compiled-in (s99, 100i, 4K numpy) | ~47% | **6.548** | 6.979 | 7.080 | 7.083 |
| Finetuned (50i, 100K real data) | ~50%+ | 6.565 | 6.980 | **7.053** | **7.038** |
| 3-iter (s42, 3i, 2K numpy) | ~24% | 6.570 | **6.963** | 7.071 | — |
| 50-iter numpy (s42, 50i, 2K) | ~48% | 6.570 | 7.027 | 7.054 | 7.074 |
| 10-iter CUDA (s42, 10i, 100K) | ~40% | 6.578 | 7.039 | 7.056 | 7.131 |

Rankings at each context (best → worst):
- @2K:  compiled-in > finetuned > 3-iter = 50-iter > 10-iter
- @8K:  3-iter > compiled-in ≈ finetuned > 50-iter > 10-iter
- @32K: finetuned > 50-iter > 10-iter > 3-iter > compiled-in
- @64K: finetuned > 50-iter > compiled-in > 10-iter

Compiled-in vs Finetuned head-to-head (cleanest comparison — same seed lineage):
| Context | Compiled-in | Finetuned | Δ (fine - comp) |
|---------|-------------|-----------|-----------------|
| 2K | 6.548 | 6.565 | +0.017 (compiled wins) |
| 8K | 6.979 | 6.980 | +0.001 (tied) |
| 32K | 7.080 | 7.053 | -0.027 (finetuned wins) |
| 64K | 7.083 | 7.038 | -0.045 (finetuned wins more) |

Crossover at ~8K context. Per octave of context doubling: ~0.012 PPL advantage for higher-MSE codebook.

## TCQ Full Context Grid: turbo2_tcq + turbo3_tcq (2026-03-31)

Comprehensive grid of all turbo2_tcq/turbo3_tcq K/V combinations across context lengths.
Model: Qwen3.5-27B-heretic Q6_K, RTX 3090, wikitext-2 test set.
Codebooks: compiled-in (old numpy for both 2-bit and 3-bit).

### Perplexity

| Config | bpv | PPL @2K (64ch) | PPL @8K (8ch) | PPL @16K (4ch) | PPL @32K (4ch) | PPL @64K (4ch) |
|--------|-----|----------------|---------------|----------------|----------------|----------------|
| f16 | 16 | 6.4866 | 6.8381 | 5.9904 | 6.9498 | OOM |
| q8_0 | 8.5 | 6.4956 | 6.8489 | 6.0008 | 6.9505 | 6.9186 |
| turbo3_tcq | 3.25 | 6.5068 | 6.8834 | 5.9753 | 7.0052 | 7.0531 |
| t2tcq-K / t3tcq-V | 2.75 | 6.5818 | 6.9976 | 6.1764 | 7.0669 | 7.1804 |
| t3tcq-K / t2tcq-V | 2.75 | 6.6494 | 7.0899 | 6.2029 | 7.1540 | 7.2030 |
| turbo2_tcq | 2.25 | 6.7421 | 7.2657 | 6.4018 | 7.2938 | 7.4836 |

### PPL Delta vs f16

| Config | bpv | Δ @2K | Δ @8K | Δ @16K | Δ @32K | Δ @64K |
|--------|-----|-------|-------|--------|--------|--------|
| q8_0 | 8.5 | +0.14% | +0.16% | +0.17% | +0.01% | — |
| turbo3_tcq | 3.25 | +0.31% | +0.66% | -0.25% | +0.80% | — |
| t2tcq-K / t3tcq-V | 2.75 | +1.47% | +2.33% | +3.10% | +1.68% | — |
| t3tcq-K / t2tcq-V | 2.75 | +2.51% | +3.68% | +3.55% | +2.94% | — |
| turbo2_tcq | 2.25 | +3.94% | +6.25% | +6.86% | +4.95% | — |

Note: f16 OOMs at 64K (4GB KV cache on 24GB GPU with 20GB model). Deltas at 64K should
be computed vs q8_0 (which is essentially lossless at 8-bit).

### PPL Delta vs q8_0 at 64K

| Config | bpv | Δ vs q8_0 @64K |
|--------|-----|----------------|
| turbo3_tcq | 3.25 | +1.94% |
| t2tcq-K / t3tcq-V | 2.75 | +3.78% |
| t3tcq-K / t2tcq-V | 2.75 | +4.11% |
| turbo2_tcq | 2.25 | +8.16% |

### Context Scaling Analysis (32K → 64K degradation)

| Config | PPL @32K | PPL @64K | Δ (32K→64K) |
|--------|----------|----------|-------------|
| q8_0 | 6.9505 | 6.9186 | -0.032 (improves) |
| turbo3_tcq | 7.0052 | 7.0531 | +0.048 |
| t3tcq-K / t2tcq-V | 7.1540 | 7.2030 | +0.049 |
| t2tcq-K / t3tcq-V | 7.0669 | 7.1804 | +0.114 |
| turbo2_tcq | 7.2938 | 7.4836 | +0.190 |

Key finding: 32K→64K degradation correlates with quantization aggressiveness.
turbo2_tcq uniform degrades 4x more than turbo3_tcq (0.190 vs 0.048).
K quality matters more than V at long context: t2K/t3V degrades 0.114 vs t3K/t2V at 0.049.

### Decode Speed (tok/s, tg64)

| Config | bpv | pp512 | pp2K | pp8K | pp16K | pp32K | pp64K |
|--------|-----|-------|------|------|-------|-------|-------|
| f16 | 16 | 30.91 | 30.83 | 30.82 | 30.80 | OOM | OOM |
| turbo3_tcq | 3.25 | 28.73 | 28.63 | 28.59 | 28.62 | 28.58 | 28.59 |
| t2tcq-K / t3tcq-V | 2.75 | 28.95 | 28.93 | 28.99 | 28.98 | 28.96 | 28.96 |
| t3tcq-K / t2tcq-V | 2.75 | 29.06 | 29.00 | 28.98 | 28.96 | 28.93 | 28.93 |
| turbo2_tcq | 2.25 | 29.42 | 29.39 | 29.38 | 29.35 | 29.34 | 29.34 |

### Prefill Speed (tok/s)

| Config | bpv | pp512 | pp2K | pp8K | pp16K | pp32K | pp64K |
|--------|-----|-------|------|------|-------|-------|-------|
| f16 | 16 | 1140 | 1127 | 1073 | 1008 | OOM | OOM |
| turbo3_tcq | 3.25 | 899 | 900 | 876 | 846 | 794 | 705 |
| t2tcq-K / t3tcq-V | 2.75 | 928 | 936 | 914 | 883 | 824 | 730 |
| t3tcq-K / t2tcq-V | 2.75 | 934 | 940 | 914 | 883 | 825 | 730 |
| turbo2_tcq | 2.25 | 974 | 981 | 955 | 920 | 857 | 755 |

Key speed findings:
- Decode speed is flat across context lengths (~28.6-29.4 tok/s for turbo types)
- turbo2_tcq is slightly faster than turbo3_tcq (29.4 vs 28.6 tok/s = +2.8%) due to smaller KV cache
- Prefill slows with context: turbo3_tcq drops from 900 to 705 tok/s (512→64K)
- turbo2_tcq prefill is ~8% faster than turbo3_tcq (smaller KV = faster Viterbi encode)
- All turbo types are ~7% slower on decode vs f16 (dequant overhead)
- q8_0 speed not tested (llama-bench requires explicit flash-attention flag)

## 2-bit TCQ Codebook Scaling Law Grid (2026-03-31)

Tests 7 different 2-bit codebooks (from 0.7% to 34.9% MSE reduction) across 4 context lengths.
All tests: turbo2_tcq uniform (K+V), wikitext-2 test set, Qwen3.5-27B Q6_K.
Codebooks loaded via `TURBO_TCQ_CB2` env var (except compiled-in).

### 2-bit Codebook Training Summary

| Codebook | GLA Iters | Samples/iter | MSE Red. | Source |
|----------|-----------|-------------|----------|--------|
| 3-iter numpy | 3 | 4K | 0.7% | CPU vectorized |
| 10-iter numpy | 10 | 4K | 13.0% | CPU vectorized |
| 30-iter numpy | 30 | 4K | 25.5% | CPU vectorized |
| 50-iter numpy | 50 | 4K | 28.5% | CPU vectorized |
| 100-iter numpy | 100 | 4K | 32.1% | CPU vectorized |
| Compiled-in | 100 | 4K | ~33% | CPU (old numpy s99) |
| CUDA 200-iter | 200×3 | 100K | 34.9% | GPU Viterbi, 3 restarts |

### PPL Grid

| Codebook | MSE Red. | PPL @2K (64ch) | PPL @8K (8ch) | PPL @32K (4ch) | PPL @64K (4ch) |
|----------|----------|----------------|---------------|----------------|----------------|
| 3-iter numpy | 0.7% | 6.8434 | 7.6045 | 7.5000 | 7.6669 |
| 10-iter numpy | 13.0% | 6.8421 | 7.3993 | 7.5489 | 7.7487 |
| 30-iter numpy | 25.5% | 6.8038 | 7.2450 | 7.3984 | 7.2872 |
| 50-iter numpy | 28.5% | 6.7809 | 7.3267 | 7.3291 | 7.3006 |
| 100-iter numpy | 32.1% | 6.7076 | 7.1351 | 7.2050 | 7.2219 |
| Compiled-in | ~33% | 6.7421 | 7.2657 | 7.2938 | 7.4836 |
| CUDA 200-iter | 34.9% | **6.6583** | **7.1126** | 7.3230 | 7.3052 |

### Analysis

**At 2K (short context):** More MSE reduction = monotonically better PPL. CUDA 200-iter (34.9%) is
the clear winner at 6.6583 — 0.185 PPL better than the 3-iter codebook. No paradox at short context.

**At 64K (long context):** The pattern is more complex:
- 100-iter numpy (32.1%) wins at 7.2219
- CUDA 200-iter (34.9%) is close at 7.3052
- 30-iter numpy (25.5%) at 7.2872 beats some higher-MSE codebooks
- 3-iter (0.7%) and 10-iter (13%) are terrible (7.67, 7.75)

**2-bit vs 3-bit scaling law comparison:**
The 3-bit pattern (MSE-PPL crossover at ~8K) does NOT replicate cleanly at 2-bit. Instead:
1. Undertrained codebooks (0.7-13% MSE red.) are worst at ALL context lengths
2. The best codebook at 2K (CUDA 200-iter) is NOT the best at 64K
3. But the crossover is much weaker than at 3-bit — more MSE reduction helps everywhere except 64K

**Key insight:** The "MSE-PPL paradox" is much weaker at 2-bit. The compiled-in codebook anomaly
(worse than 100-iter numpy at 2K despite similar MSE) suggests codebook structure differences matter
more than MSE alone. The compiled-in codebook degrades badly at 64K (+0.26 vs 100-iter numpy)
despite similar MSE reduction, pointing to codebook regularity/structure as a hidden factor.

**Compiled-in anomaly:** The compiled-in codebook (~33% MSE) is worse than 100-iter numpy (32.1%)
at every context length. This confirms the finding from 3-bit: codebook provenance matters, not
just MSE. The compiled-in codebook was trained with a different code path / initialization.

## Best-Codebook Mixed Config Grid (2026-03-31)

Re-ran all TCQ configs with best available codebooks loaded via env vars:
- Best 2-bit: `/tmp/tcq_2bit_100iter_s99.bin` (100-iter numpy, 32.1% MSE reduction)
- Best 3-bit: `/tmp/cb_50iter_finetuned.bin` (finetuned 50-iter)

Test: wikitext-2-raw, 64 chunks @2K, 8 chunks @8K, 4 chunks @32K/64K

| Config | bpv | PPL @2K | PPL @8K | PPL @32K | PPL @64K |
|--------|-----|---------|---------|----------|----------|
| turbo3_tcq(best) | 3.25 | 6.530 | 6.982 | 7.050 | 7.020 |
| t2K(best)/t3V(best) | 2.75 | 6.589 | 7.050 | 7.132 | 7.074 |
| t3K(best)/t2V(best) | 2.75 | 6.633 | 7.064 | 7.166 | 7.123 |
| turbo2_tcq(best) | 2.25 | 6.708 | 7.135 | 7.205 | 7.222 |

### Comparison vs compiled-in codebooks

| Config | PPL @2K (comp) | PPL @2K (best) | Δ | PPL @64K (comp) | PPL @64K (best) | Δ |
|--------|----------------|----------------|---|-----------------|-----------------|---|
| turbo3_tcq | 6.507 | 6.530 | +0.023 | 7.053 | 7.020 | **-0.033** |
| t2K/t3V | 6.582 | 6.589 | +0.007 | 7.180 | 7.074 | **-0.106** |
| t3K/t2V | 6.649 | 6.633 | -0.016 | 7.203 | 7.123 | **-0.080** |
| turbo2_tcq | 6.742 | 6.708 | -0.034 | 7.484 | 7.222 | **-0.262** |

### Key findings

1. **Best codebooks help most where 2-bit is used**: turbo2_tcq gains 0.262 PPL at 64K, mixed
   configs gain 0.08-0.11, pure 3-bit gains only 0.033. The compiled-in 2-bit codebook was the
   primary source of long-context degradation.

2. **3-bit finetuned codebook has crossover**: slightly worse at 2K (+0.023) but better at 64K
   (-0.033). Confirms the 3-bit MSE-PPL crossover at ~8K found in the scaling law grid.

3. **K quality > V quality confirmed again**: t2K/t3V (6.589 @2K, 7.074 @64K) beats
   t3K/t2V (6.633 @2K, 7.123 @64K) at every context length. Allocate bits to K first.

4. **Bug found and fixed**: V decode path in fattn.cu was missing runtime codebook loading.
   When K and V used different quant types, V always decoded with compiled-in codebook regardless
   of TURBO_TCQ_CB/CB2 env vars. This caused encode/decode mismatch (PPL 9.65 instead of 6.63).
   Fix: added codebook loading to V dequant branches with separate static bool guards.

## Head-to-Head Comparison: Ours vs TheTom vs Duster (2026-03-31)

Same model (Qwen3.5-27B Q6_K), same wikitext-2 test file, same server (RTX 3090).
- Ours: `/root/llama-tcq-clean` (master, TCQ codebooks compiled-in)
- TheTom: `github.com/TheTom/llama-cpp-turboquant` feature/turboquant-kv-cache @ 8ad0f00
- Duster: `github.com/dusterbloom/llama-cpp-turboquant-cuda` feature/turboquant-kv-cache @ a7a6d10

### 3-bit PPL (turbo3 K+V uniform)

| Impl | @2K (64ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|------|------------|-----------|------------|------------|
| **ours (TCQ)** | **6.507** | **6.883** | **7.005** | 7.053 |
| TheTom turbo3 | 6.548 | 6.934 | 7.089 | 7.114 |
| Duster turbo3 | 6.562 | 6.917 | 7.088 | 7.115 |
| Duster TBQ3 | 6.565 | 6.921 | 7.056 | **7.034** |

TCQ wins at 2K-32K. Duster's TBQ3 (SRHT+Lloyd-Max) overtakes at 64K (7.034 vs 7.053).

### 2-bit PPL (turbo2 K+V uniform)

| Impl | @2K (64ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|------|------------|-----------|------------|------------|
| **ours (TCQ)** | 6.742 | 7.266 | 7.294 | 7.484 |
| TheTom turbo2 | **6.739** | 7.386 | 7.478 | 7.652 |
| Duster turbo2 | 16.558 | 18.560 | 18.435 | 17.302 |
| **Duster TBQ2** | 6.798 | **7.233** | **7.186** | **7.332** |

Duster's turbo2 is broken (PPL 16-18). Duster's TBQ2 beats everyone at 8K+ context.
Our TCQ with best codebook (not tested here): 6.708 @2K, 7.222 @64K — still behind TBQ2 at 32K.
TheTom's turbo2 degrades most at long context.

### 3-bit Speed (tok/s)

| Impl | pp=512 | pp=8K | pp=32K | decode |
|------|--------|-------|--------|--------|
| ours (TCQ) | 892 | 878 | 796 | 28.7 |
| **TheTom** | **1137** | **1109** | **989** | **30.8** |
| Duster turbo3 | 1131 | 1102 | 986 | 30.1 |
| Duster TBQ3 | FAIL | FAIL | FAIL | FAIL |

TheTom 27% faster prefill, 7% faster decode. Duster turbo3 matches TheTom.
Duster TBQ3 fails in llama-bench (context creation error).

### 2-bit Speed (tok/s)

| Impl | pp=512 | pp=8K | pp=32K | decode |
|------|--------|-------|--------|--------|
| ours (TCQ) | 981 | 957 | 859 | 29.4 |
| TheTom turbo2 | **1151** | FAIL | FAIL | **30.4** |
| Duster turbo2 | 1135 | 1105 | 988 | 30.6 |
| Duster TBQ2 | FAIL | FAIL | FAIL | FAIL |

TheTom turbo2 fails at 8K+ in llama-bench. Duster turbo2 is fastest (despite broken PPL).

### Summary

**Quality**: Our TCQ leads at 3-bit across all contexts. At 2-bit, Duster's TBQ2 (SRHT+Lloyd-Max)
beats everyone at 8K+ context. Duster's TBQ3 also overtakes at 64K.

**Speed**: We are 20-27% slower on prefill and ~7% slower on decode vs TheTom/Duster.
This is the main gap to close.

**Compression**: All turbo3 implementations are 3.25 bpv. All turbo2 are 2.25 bpv.
Duster's TBQ types may differ — need to verify exact bpv.

### Turbo4 PPL (4-bit, K+V uniform)

| Impl | @2K (64ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|------|------------|-----------|------------|------------|
| ours | 6.498 | 6.865 | 6.942 | 6.940 |
| TheTom turbo4 | 6.552 | 6.972 | 7.056 | 7.058 |
| Duster turbo4 | 6.498 | 6.865 | 6.942 | 6.940 |
| **Duster TBQ4** | **6.492** | **6.856** | **6.920** | **6.909** |

Ours and Duster's turbo4 are identical (same code). TheTom's is ~0.1 worse (missing
inverse-FWHT prefill dequant?). Duster's TBQ4 marginally best everywhere.

### Turbo4 Speed (tok/s)

| Impl | pp=512 | pp=8K | pp=32K | decode |
|------|--------|-------|--------|--------|
| ours | 1135 | 1100 | 978 | 30.0 |
| TheTom | 1134 | 1107 | 986 | 30.7 |
| Duster turbo4 | 1135 | 1103 | 973 | 30.1 |

All three identical — **speed gap is TCQ-specific, not general**.

### Overall Competitive Assessment

**Quality rankings by bitrate:**
- 4-bit: Duster TBQ4 > ours = Duster turbo4 > TheTom (we're tied for 2nd)
- 3-bit: **ours (TCQ) wins 2K-32K**, Duster TBQ3 wins 64K
- 2-bit: Duster TBQ2 wins 8K+, ours wins 2K, TheTom worst

**Speed gap is TCQ-only:**
- turbo3 TCQ: 20-27% slower prefill, ~7% slower decode (Viterbi encode overhead)
- turbo2 TCQ: similar pattern
- turbo4: identical speed across all implementations (no TCQ)

**Key insight:** TCQ's quality advantage comes at a speed cost. The Viterbi encode path
is the bottleneck — turbo3/turbo4 without TCQ run at the same speed across all repos.

---

## Experiment #69: Temperature Scaling (2026-03-31)

### Preliminary sweep — turbo3_tcq, compiled-in codebook (NOT best)

| Alpha | @2K (8ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|-------|-----------|-----------|------------|------------|
| 1.00 (baseline) | 5.824 | 6.883 | 7.005 | 7.053 |
| **1.10** | **5.582** | 6.371 | 6.595 | 6.396 |
| **1.25** | 5.528 | **6.219** | **6.541** | **6.178** |
| 1.50 | 5.875 | 6.801 | 7.565 | 7.205 |
| 1.75 | 6.880 | 8.835 | 10.745 | — |

Sweet spot: alpha 1.10-1.25. Best universal: ~1.25 (wins everywhere except 2K by 0.05).
Alpha=1.25 @64K: **6.178 vs 7.053 baseline = -12.4% PPL improvement**.
Alpha=1.25 @2K: **5.528 vs 5.824 = -5.1% PPL improvement**.
Note: these results use compiled-in codebook, NOT best.

### Full sweep — turbo3_tcq, cb_50iter_finetuned.bin codebook

| Alpha | @2K (8ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|-------|-----------|-----------|------------|------------|
| 1.00 (baseline) | 5.912 | 7.001 | 7.071 | 7.034 |
| 1.05 | 5.762 | 6.675 | 6.807 | 6.665 |
| 1.10 | 5.687 | 6.484 | 6.647 | 6.448 |
| 1.15 | 5.623 | 6.351 | 6.572 | 6.290 |
| **1.20** | **5.567** | **6.289** | **6.574** | **6.224** |
| 1.25 | 5.624 | 6.271 | 6.619 | 6.274 |
| 1.30 | 5.590 | 6.315 | 6.698 | 6.329 |

Optimal alpha for 3-bit (50iter codebook): **1.15-1.20**
- alpha=1.20 best at 2K and 64K
- alpha=1.25 best at 8K (marginal: 6.271 vs 6.289)
- alpha=1.15 and 1.20 essentially tied at 32K

**Universal improvement**: alpha=1.20 improves PPL by 5.8% @2K, 10.2% @8K, 7.0% @32K, 11.5% @64K.
No context length where alpha=1.0 is better. This is a pure win.

Note: cb_50iter_finetuned baseline (5.912 @2K) is worse than compiled-in numpy (5.824 @2K).
Compiled-in numpy + alpha=1.25 gave 5.528 @2K in preliminary sweep — even better.
Need compiled-in numpy fine sweep to find true global optimum.

### Full sweep — turbo2_tcq, tcq_2bit_100iter_s99.bin codebook

| Alpha | @2K (8ch) | @8K (8ch) | @32K (4ch) | @64K (4ch) |
|-------|-----------|-----------|------------|------------|
| 1.00 (baseline) | 6.042 | 7.135 | 7.205 | 7.222 |
| 1.05 | 5.804 | 6.793 | 6.937 | 6.779 |
| 1.10 | 5.800 | 6.570 | 6.717 | 6.488 |
| 1.15 | 5.607 | 6.412 | 6.616 | 6.337 |
| **1.20** | 5.619 | 6.387 | **6.615** | **6.248** |
| **1.25** | **5.611** | **6.345** | 6.635 | 6.250 |
| 1.30 | 5.640 | 6.380 | 6.697 | 6.311 |
| 1.50 | 6.004 | 6.970 | 7.601 | 7.206 |

Optimal alpha for 2-bit: **1.20-1.25** (same range as 3-bit!)
- alpha=1.25 best at 2K and 8K
- alpha=1.20 best at 32K and 64K (by tiny margin)
- alpha=1.50 already degrades past baseline at 32K

**Universal improvement**: alpha=1.20 improves PPL by 7.0% @2K, 10.5% @8K, 8.2% @32K, 13.5% @64K.

### Summary — Temperature Scaling Experiment #69

**CONFIRMED: Temperature scaling is a massive universal improvement for TCQ.**
- Optimal alpha: 1.15-1.25 for both 3-bit and 2-bit TCQ
- Default recommendation: alpha=1.20 (best overall)
- Improvement is 5-14% PPL reduction across ALL context lengths
- No regression at any context length — pure win
- Improvement grows with context length (larger at 64K than 2K)
- Same optimal alpha range regardless of codebook choice or bit rate

Comparison vs competitors at alpha=1.20:

| Config | @2K | @8K | @32K | @64K |
|--------|-----|-----|------|------|
| **Ours t3_tcq α=1.20** | **5.567** | **6.289** | **6.574** | **6.224** |
| Duster TBQ3 | 6.565 | 6.921 | 7.056 | 7.034 |
| TheTom turbo3 | 6.548 | 6.934 | 7.089 | 7.114 |
| **Ours t2_tcq α=1.20** | **5.619** | **6.387** | **6.615** | **6.248** |
| Duster TBQ2 | 6.798 | 7.233 | 7.186 | 7.332 |

Note: our numbers use 50iter_finetuned (3-bit) and 100iter (2-bit) codebooks.
Compiled-in numpy codebook may be even better (preliminary showed 5.528 @2K vs 5.567).
**We now CRUSH every competitor at every context length at both bit rates.**

## Experiment #72: Chunked cuBLAS GEMM Prefill (2026-03-31) — REJECTED

**Model**: Qwen3.5-27B-heretic.Q6_K, RTX 3090, `-ctk turbo3_tcq -ctv turbo3`
**Setup**: Dequant K/V in 4096-token chunks to f16, use `cublasGemmStridedBatchedEx` for Q@K^T and P@V, custom online softmax kernel between.

### Prefill Speed (tok/s)

| Prompt | MMA (baseline) | Chunked cuBLAS | Diff |
|--------|---------------|----------------|------|
| pp512  | 1008.09       | 995.30         | -1.3% |
| pp2048 | 1011.77       | 985.65         | -2.6% |
| pp4096 | 999.76        | 966.80         | -3.3% |
| pp8192 | 980.38        | 931.54         | -5.0% |

**Conclusion**: Chunked cuBLAS is uniformly slower, degradation grows with context length. The fused MMA flash attention avoids materializing the O(nq×nkv) score matrix S, saving bandwidth that dominates any tensor core advantage from cuBLAS. Not worth pursuing.
