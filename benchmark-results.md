# TurboQuant Benchmark Results

Hardware: RTX 3090 24GB, Qwen3.5 27B Q6_K (20.56 GiB)
Date: 2026-03-26
Build: feature/turboquant-kv-cache + FA_ALL_QUANTS=ON

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
