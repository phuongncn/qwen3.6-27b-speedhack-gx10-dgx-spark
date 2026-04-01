# TurboQuant CUDA Experiments

Tracking optimization ideas, external research, and benchmark results.
Status: `done` | `ready` (can implement now) | `needs-research` | `blocked` | `dropped`

## Baseline (Qwen3.5 27B Q6_K, RTX 3090)

```
PPL (2K ctx, 8 chunks):
  q8_0:   5.8375
  turbo3: 5.8323  (-0.09%)
  turbo4: 5.8186  (-0.32%)

Decode speed tg64 (tok/s):
  CTX       q8_0    turbo3   turbo4   t3/q8   t4/q8
  4K       31.02    29.93    29.43    0.965   0.949
  16K      30.77    29.65    29.41    0.964   0.956
  32K      30.69    29.83    29.47    0.972   0.960

Prefill pp4096 (tok/s):
  q8_0:   1134.64
  turbo3:  631.09  (0.556x)
  turbo4:  586.71  (0.517x)
```

---

## Done

### 1. Register centroid LUT
**Status**: done
**Type**: speed
**Result**: Eliminated constant memory serialization in FA inner loop. Precompute `centroid[i] * norm` in float registers. Fixed the context scaling regression TheTom was debugging.

### 2. Batch uint32_t 3-bit unpack (turbo4)
**Status**: done
**Type**: speed
**Result**: Single 32-bit load for 8 elements instead of per-element byte manipulation.

### 3. V_DOT2 half2 accumulation path
**Status**: done
**Type**: speed (AMD)
**Result**: On AMD GPUs with `v_dot2_f32_f16`, accumulate K dot products using half2 pairs via `ggml_cuda_mad`.

### 4. turbo4 V dequant optimization
**Status**: done
**Type**: speed
**Result**: Register LUT + batch qs/signs loads for V dequantization.

### 5. Norm correction (turbo3 + turbo4)
**Status**: done
**Type**: quality (zero decode cost)
**Result**: Store `original_norm / ||reconstruction||` instead of `original_norm`. turbo4 PPL now *beats* q8_0 (5.8186 vs 5.8375).

### 6. fp16 centroid LUT (TheTom, upstream 654647aac)
**Status**: done
**Type**: speed
**Result**: +6-14% decode at long context. Superseded by our register LUT.

### 7. Float norm broadcast (TheTom, upstream aa6a3a180)
**Status**: done
**Type**: speed
**Result**: +2-3% decode over fp16 LUT.

---

## Ready to Test

### 8. Layer-adaptive mode 2 (last 8 layers q8_0)
**Status**: done — **validated**
**Type**: quality + speed
**Result**: LA-2 turbo3: PPL 5.8140 (-0.40%), 97.7% decode speed. LA-2 turbo4: PPL 5.8077 (-0.51%), 96.7% decode speed. Both beat uniform turbo AND q8_0 in quality. Matches Tom's findings. **LA-2 turbo3 is the recommended config** (best quality/compression/speed balance).

### 9. Layer-adaptive mode 1 (first 4 + last 4 q8_0)
**Status**: done — **best PPL tested!**
**Type**: quality
**Result**: LA-1 turbo3: PPL 5.7958 (-0.71% vs q8_0), 97.7% decode speed. Beats LA-2 turbo3 (5.8140) by 0.31%. Protecting BOTH early residual stream and final output layers is better than just the last 8. Same compression ratio and speed as LA-2 since both use 8 q8_0 layers.

### 10. Asymmetric K/V combinations
**Status**: done — **mixed results**
**Type**: quality/speed
**Results**:
  - turbo4-K + q8_0-V: PPL 5.8451 (+0.13%), 98.2% speed — fast but slightly worse than q8_0
  - q8_0-K + turbo3-V: PPL 5.8451 (+0.13%), 98.8% speed — fastest config tested
  - turbo4-K + turbo3-V: PPL 5.8653 (+0.48%) — worst combo
  - turbo3-K + turbo4-V: PPL 5.8212 (-0.28%) — good! Values need more precision
**Surprise**: "More for Keys, Less for Values" paper prediction was WRONG for this model. turbo3-K + turbo4-V beats turbo4-K + turbo3-V by 0.76% PPL. Values matter more on Qwen3.5 27B.
**Note**: All asymmetric turbo+q8 combos have slightly worse PPL than pure q8_0. The norm correction gives uniform turbo an edge that mixing with uncorrected q8_0 dilutes.

### 11. Layer-adaptive + asymmetric combined
**Status**: done — **NEGATIVE RESULT**
**Type**: quality
**Branch**: `experiment/attention-sink-protection`
**What**: Decouple K/V in the adaptive logic. Since experiment 10 showed values matter more than keys on Qwen3.5 27B, tested promoting only V or only K to q8_0 on sensitive layers.
**Results**:
  - Mode 6 (V-only q8_0 last 8): PPL 5.8390 (+0.03%) — WORSE than uniform turbo3
  - Mode 7 (K-only q8_0 last 8): PPL 5.8390 (+0.03%) — identical to mode 6
  - Mode 8 (V-only q8_0 first2+last2): PPL 5.8330 (-0.08%) — ~= uniform
**Finding**: Promoting only one of K/V hurts quality due to norm correction mismatch between turbo and q8_0 within the same layer. K vs V makes no difference. Both must be promoted together (mode 2: 5.8140) for the quality improvement to work.

### 11b. Layer-adaptive modes 3, 4, 5 — isolation tests
**Status**: done
**Type**: quality
**Results** (all turbo3):
  - Mode 3 (last 4 only): PPL 5.8091 (-0.49%), 4 layers q8_0, ~4.2x compression
  - Mode 4 (first 4 only): PPL 5.8211 (-0.28%), 4 layers q8_0, ~4.2x compression
  - Mode 5 (first 2 + last 2): PPL 5.8091 (-0.49%), 4 layers q8_0, ~4.2x compression
**Key insight**: Mode 3 = Mode 5 (same PPL). The last 2 layers are the critical ones — protecting them dominates. The first 4 layers contribute less than the last 4. Mode 5 is the sweet spot for max compression: only 4 layers q8_0 yet still beats q8_0 by 0.49%.

### 11e. Extreme context test (65K+)
**Status**: done
**Type**: VRAM / speed
**Result**: All turbo configs fit at 65K on 24GB RTX 3090 (~22.2-22.3 GiB). Decode speed at 65K is virtually identical to 32K — zero degradation. LA-1 turbo3: 29.98 tok/s, LA-5 turbo3: 29.90, turbo4: 29.51. q8_0 would OOM at this context length (~28+ GiB needed).

---

### 16. Prefill dequant-then-attend (dequant to fp16 + MMA)
**Status**: done — **turbo3 + turbo4**
**Type**: speed (prefill)
**Branch**: `experiment/prefill-dequant-attend`
**Result**: turbo3 prefill 631→1125 tok/s (1.78x, 98.8% of q8_0). turbo4 prefill 588→1113 tok/s (1.9x, 98.1% of q8_0).
**turbo4 note**: QJL correction (~0.001 magnitude) rounds away in fp16 temp buffer. turbo4 prefill PPL 5.8966 vs 5.8186 full precision (+1.3%). Accepted tradeoff: only prompt tokens affected, generated tokens use full-precision SET_ROWS.
**Bug fixed**: turbo4 dequant_f16 kernel had missing block indexing for ne0 > QK_TURBO4 (Qwen3.5-27B has head_dim=256).

### 16b. turbo4 prefill — accepted fp16 tradeoff
**Status**: done — **ENABLED, 1.9x prefill speedup**
**Type**: speed (prefill, turbo4 only)
**What**: Enabled fp16 dequant + MMA for turbo4 prefill. QJL loses ~1% PPL precision in fp16 round-trip, but 2x prefill speedup is worth it since only prompt tokens are affected.
**Result**: turbo4 pp4096 = 1113 tok/s (was 588). PPL 5.8966 (all-prefill worst case). Real inference quality between 5.82-5.90 depending on prompt/generation ratio.

---

## Needs Research — Prefill Speed (SOLVED: 98.8% of q8_0, deprioritized)

### 12. BitDecoding-style MMA kernel with dequant pipelining
**Status**: needs-research
**Type**: speed (prefill)
**Paper**: BitDecoding (HPCA 2026, arXiv:2503.18773), open source at github.com/OpenBitSys/BitDecoding
**What**: First system using Tensor Cores for low-bit KV cache decoding. Register-level software pipeline: while tensor cores execute `mma.sync` on tile N, CUDA cores dequant tile N+1. Drops dequant overhead from 40-50% to 15%. Uses `lop3` PTX for bit manipulation + `ldmatrix` for TC layout.
**Performance**: 7.5x on RTX4090, 4.8x on A100, 8.9x on H100 vs FP16.
**How it applies**: Turbo dequant (bit extract + LUT + norm multiply) is pure ALU — ideal for overlapping with MMA. Turbo3's split qs+signs layout maps well to `lop3`. This is the most promising path to fix our prefill gap.
**Difficulty**: High (3-4 weeks). Need to restructure fattn-mma to add dequant pipeline stage.

### 13. SageAttention INT8 intermediate path
**Status**: needs-research
**Type**: speed (prefill)
**Paper**: SageAttention (ICLR 2025), SageAttention2 (ICML 2025), github.com/thu-ml/SageAttention
**What**: Quantize Q,K to INT8 before attention matmul, use INT8 tensor cores (`mma u8.u8.s32`) which have 2x throughput of FP16 MMA on Ampere. K smoothing (subtract mean) exploits softmax shift-invariance.
**How it applies**: Instead of dequanting turbo3/4 to FP16 for MMA, dequant to INT8 and use INT8 TC. Path: load turbo block → bitfield extract → codebook lookup → quantize to INT8 → feed INT8 MMA.
**Risk**: Double-quantization (turbo → INT8) may accumulate error. Need PPL validation.
**Difficulty**: Medium-High (2-3 weeks). Could make prefill *faster* than q8_0.

### 14. TurboMind 3-stage software pipeline
**Status**: needs-research
**Type**: speed (prefill + decode)
**Paper**: LMDeploy TurboMind (arXiv:2508.15601)
**What**: Explicitly overlaps 3 stages: (1) TC execute `mma.sync` on current tile, (2) INT/FP ALU dequant next tile, (3) `cp.async` prefetch subsequent tile. 61% latency reduction, 156% throughput improvement.
**How it applies**: Same principle as BitDecoding but more explicitly structured. Our fattn-mma already uses `cp.async` — the missing piece is inserting a dequant stage between load and MMA.
**Difficulty**: Medium-High (2-3 weeks).

### 15. Shared memory KV block caching (prefill)
**Status**: needs-research
**Type**: speed (prefill)
**What**: During prefill, multiple query tokens access the same KV positions. Dequantize a KV block once into shared memory, all query threads read from it.
**Challenge**: Shared memory is ~48KB/SM. A turbo4 block = 128 floats = 512 bytes dequantized. Balance cache size vs occupancy.
**Difficulty**: Medium (1-2 weeks).

### 16 (original). ~~Prefill-specific dequant-then-attend~~
**Status**: done — **superseded by #16 above** (turbo3 1125 tok/s, 98.8% of q8_0)

---

## Needs Research — Decode Speed (polish: 95-97% → parity)

### 17. Split-K / FlashDecoding tuning for turbo decode
**Status**: done — **NO EFFECT** (see #49)
**Type**: speed (decode)
**Papers**: FlashDecoding (Stanford), FlashDecoding++ (MLSys 2024)
**Result**: Tested via #49 (GGML_PARALLEL_BLOCKS override). All parallel_blocks values 1-32 produce identical decode speed (~29.95 tok/s). Attention is <5% of decode time — FFN dominates. The remaining 2.8% turbo3→q8_0 gap is structural and can't be closed by attention tuning. Items 3-5 (nbatch_fa, stream_k, async softmax) also won't help since attention isn't the bottleneck.

### 18. SAS softmax optimization
**Status**: dropped — **attention <5% of decode, prefill already 98.8%**
**Type**: speed (both)
**Paper**: TurboAttention (Microsoft, arXiv:2412.08585)
**What**: Decompose `exp(-x) = LUT(-x_int) * polynomial(-x_dec)`, polynomial runs on tensor cores in FP16.
**Finding**: 5-15% attention speedup × <5% attention share = <0.75% total throughput gain. Not worth the complexity. Prefill already at 98.8% of q8_0. Only revisit if targeting attention-heavy workloads (very long context decode on MoE models).

### 25b. Sign+magnitude encoding for turbo3 dequant
**Status**: done — **NEUTRAL** (no measurable speedup)
**Type**: speed (decode)
**Branch**: `experiment/sign-magnitude-encoding`
**What**: Remap turbo3's 3-bit index from {low2, high1} → {mag_idx, sign_bit}. Dequant uses 4-entry magnitude LUT + conditional negate instead of 8-entry centroid LUT. Halves register LUT pressure.
**Results**:
  - PPL: 5.8501 (identical to baseline with MMA prefill)
  - Decode 4K: 30.05 tok/s (baseline ~30.04 = no change)
  - Decode 32K: 29.91 tok/s (baseline ~29.83 = +0.3%, within noise)
**Finding**: The decode bottleneck is memory bandwidth, not ALU/register pressure from the LUT. Halving the LUT size saves ~1 instruction per element but has no measurable impact. q8_0 is 31.03 tok/s; the 3% turbo3 gap is structural.

### 25c. Long-context PPL comparison (turbo3 vs q8_0 at 4K/8K)
**Status**: done — **quality holds at long context**
**Type**: quality validation
**Results** (all 2K/8chunks unless noted):
  - 2K: turbo3 LA-1 5.7690 (-1.17%) vs q8_0 5.8375
  - 4K/4chunks: turbo3 LA-1 6.3198 (+0.83%) vs q8_0 6.2677 (turbo3 slightly worse)
  - 8K/4chunks: turbo3 LA-1 7.3952 (-0.39%) vs q8_0 7.4241 (turbo3 wins again)
  - 8K/4chunks: turbo3 uniform 7.3783 (-0.62%) vs q8_0 7.4241
**Finding**: Quality advantage is noisy across context lengths. turbo3 generally competitive with q8_0 (±0.5%). The PPL increase at longer eval is due to wikitext data (later text harder), not degradation. Error bars (±0.16-0.18) are larger than the differences.

---

## Needs Research — Quality Improvements

### 19. Channel reordering before FWHT
**Status**: dropped — **INCOMPATIBLE with random sign arrays**
**Type**: quality (zero decode cost)
**Paper**: RotateKV (IJCAI 2025, arXiv:2501.16383)
**What**: Sort channels by magnitude so similar-magnitude channels land in the same FWHT block.
**Research** (2026-03-27): RotateKV uses deterministic grouped-head Hadamard WITHOUT random signs. Our PolarQuant uses random sign arrays (s1, s2) that already destroy all channel magnitude structure — making reordering ineffective. Same reason GSR Walsh ordering (#39) showed no benefit. The algorithm (argsort of per-channel sum across calibration data) is simple but the prerequisite (preserved magnitude structure) doesn't hold. Code at github.com/ZunhaiSu/RotateKV. MixQuant (2601.22347) has related L1-norm balancing permutation, also assumes deterministic Hadamard.
**Conclusion**: Would only help if we switched from random PolarQuant signs to deterministic Hadamard, which would lose the decorrelation benefit that makes our codebook work well.

### 20. ~~SmoothRot — channel-wise scaling before FWHT~~
**Status**: dropped — **NOT APPLICABLE to KV cache**
**Type**: quality
**Paper**: SmoothRot (arXiv:2506.05413, Jul 2025)
**Research** (2026-03-27): SmoothRot only targets **FFN down-projection** massive outliers (>100x magnitude) in GLU architectures. It does NOT target KV cache quantization or attention projections. The paper explicitly states applying smoothing before attention layers showed "limited gains." Gains also vanish when combined with GPTQ. Not applicable to our head_dim=128 KV cache quality gap.

### 21. WUSH — data-aware transform replacing pure FWHT
**Status**: needs-research — **impractical as designed, diagonal approximation viable**
**Type**: quality
**Paper**: WUSH (arXiv:2512.00956, Nov 2025, ISTA/ETH)
**What**: T_wush = H * S^{-1/2} * U^T * W'^T. Proves Hadamard is the optimal *data-agnostic* orthogonal transform, then derives optimal *data-dependent* non-orthogonal transform.
**Research** (2026-03-27): 50-60% layer loss reduction over Hadamard on K/V projections (Qwen3-8B MXFP4). End-to-end: +2.2-2.9pp quality recovery. However, the full transform is a **dense d×d matrix-vector multiply per block** — O(d²) vs O(d log d) for FWHT. With d=128: ~16384 FMAs vs ~896 for FWHT = **18x more compute**. Also requires per-model calibration + per-block matrix storage (128×128 = 64KB per block in fp16).
**Viable path**: Their Future Work mentions a **diagonal approximation** — just per-channel scaling before FWHT, O(d) cost. This is essentially what CAT (experiment #41) achieves more cleanly.
**Difficulty**: Full WUSH = impractical. Diagonal approx = see #41 (CAT alignment correction).

### 22. NSN normalization for universal codebooks
**Status**: done — **NO BENEFIT (simplified version); FULL VERSION INCOMPATIBLE**
**Type**: quality
**Paper**: NSNQuant (NeurIPS 2025, arXiv:2505.18231)
**What**: Normalize-Shift-Normalize aligns token distributions to standard normal, enabling a single reusable codebook across all layers without calibration.
**Branch**: `experiment/nsnquant-dc-removal`
**Implementation**: Simplified per-token DC removal (subtract mean, renormalize before FWHT). Full NSNQuant requires batch processing (per-channel mean across 64 tokens) which is incompatible with our per-token SET_ROWS pipeline.
**Results** (2K/8chunks):
  - turbo3 baseline: PPL 5.8501 ± 0.165
  - turbo3 + DC removal: PPL 5.8827 ± 0.166 (+0.033, noise)
  - turbo4 baseline: PPL 5.8186 ± ref
  - turbo4 + DC removal: PPL 17.4134 ± 0.618 (**catastrophic** — QJL residual breaks)
**Finding**: Per-token DC removal is useless because (1) values are already near-zero-mean after L2 normalization + FWHT, (2) for V the lost DC component corrupts the output, (3) for turbo4 the QJL sign-bit correction is computed relative to the DC-removed signal but decoded without correction. Full NSNQuant requires different infrastructure (batch quantization).

### 23. Attention-sink token protection
**Status**: done — **NO SIGNIFICANT EFFECT**
**Type**: quality
**Paper**: AnTKV (arXiv:2506.19505)
**What**: Keep first few tokens (attention sinks) at fp16 precision (pre-quantization), overwrite dequanted fp16 buffer before flash attention.
**Branch**: `experiment/attention-sink-protection`
**Results** (turbo3, 2K/8chunks):
  - No sink (baseline): PPL 5.8501 ± 0.165
  - N=4 sink tokens: PPL 5.8246 ± 0.164 (-0.026)
  - N=8 sink tokens: PPL 5.8506 ± 0.165 (+0.001)
  - N=16 sink tokens: PPL 5.8894 ± 0.167 (+0.039)
**Finding**: All deltas within error bars. turbo3 + FWHT + norm correction already has high enough quality that sink protection provides no measurable benefit. The attention-sink amplification of quantization error doesn't matter when the error is this small.
**Difficulty**: Low-Medium (1 week). High impact for chat/instruction-following quality.

### 24. Per-head adaptive precision
**Status**: deferred — **infra too heavy for uncertain gain**
**Type**: quality
**Papers**: KVC-Q (ScienceDirect 2026), KVTuner (ICML 2025), MixKVQ (arXiv:2512.19206)
**What**: Allocate turbo4 to sensitive heads, turbo3 to robust ones. Same average bit rate, better quality.
**Assessment** (2026-03-27): Requires per-head type in KV cache allocation (currently per-layer only). On Qwen3.5-27B, turbo3 is already +0.22% PPL — per-head won't help. On head_dim=128 models with only 2-4 KV heads, there's barely enough heads to differentiate. The layer-adaptive modes (#8-11) already capture most of this benefit with much less complexity.
**Difficulty**: High (2-3 weeks). Deferred until layer-adaptive is exhausted.

### 25. Drop QJL entirely (turbo3-only approach)
**Status**: done — **QJL HELPS, do NOT drop**
**Type**: simplification + speed
**Source**: TheTom's `turboquant_plus` (220 stars) + direct confirmation from Tom
**Branch**: `experiment/drop-qjl`
**What**: turbo4 uses 3-bit codebook + 1-bit QJL signs. Tom validated that dropping QJL and giving all bits to the codebook (4-bit Lloyd-Max, no QJL) is faster and equivalent quality. Block size 32 beats 128 for FA parallelism.
**Results** (turbo4 uniform, Qwen3.5-27B):
  - turbo4 WITH QJL: PPL 5.8186 (-0.32% vs q8_0), prefill 588 tok/s (vec only), decode 29.43
  - turbo4 NO QJL: PPL 5.8501 (+0.22% vs q8_0), prefill 1124 tok/s (MMA works!), decode 29.40
  - turbo3 (reference): PPL 5.8323 (-0.09% vs q8_0), prefill 1125 tok/s, decode 29.93
**Finding**: QJL contributes +0.3 PPL points to turbo4. Without QJL, turbo4 is slightly WORSE than turbo3 in quality (5.8501 vs 5.8323) and decode speed (29.40 vs 29.93), with worse compression (4.25 vs 3.5 bits/element). TheTom's finding may not apply when norm correction is in use — QJL + norm correction is what gives turbo4 its q8_0-beating quality.
**Benefit**: Dropping QJL DOES fix the fp16 prefill issue (1124 tok/s), but turbo3 already gets 1125 tok/s. No practical advantage over turbo3.
**Conclusion**: Keep QJL. turbo4's value proposition IS the QJL correction + norm correction combo. Without QJL, just use turbo3.

---

## Needs Research — Architecture / New Formats

### 26. CommVQ — RoPE-commutative codebooks
**Status**: deferred — **fundamentally different approach, not incremental**
**Type**: architecture
**Paper**: CommVQ (ICML 2025, arXiv:2506.18879, Apple/UMass)
**What**: EM-trained codebooks that commute with RoPE. Elegant but requires per-model trained codebooks and new quantization pipeline. Incompatible with our universal Lloyd-Max approach. Would be a complete rewrite, not an improvement to turbo3/4.

### 27. ~~ConvRot — group rotation instead of full-dim FWHT~~
**Status**: dropped — **failed by TheTom** (group-32 rotation: PPL 7.06 vs target 6.19)
**Paper**: ConvRot (arXiv:2512.03673, Dec 2025)
**What**: Replace full d=128 FWHT with group-of-32 Hadamard transforms. Tom tested this directly and it produces unacceptable PPL. Full d=128 rotation is necessary for proper decorrelation.

### 36. Temporal decay — progressive 3→2 bit requantization
**Status**: unblocked — turbo2 implemented (#28), needs per-position type tracking infra
**Type**: quality + memory
**Source**: TheTom/turboquant_plus/benchmarks/temporal_decay_prototype.py
**What**: Old tokens requantized turbo3→turbo2. ~30% extra memory savings.
**Prerequisites**: GGML_TYPE_TURBO2_0 now available (#28). Still needs per-position type tracking in KV cache. turbo2 benchmark data (PPL 6.78 uniform, +8% vs f16) suggests this may only be viable for tokens >16K positions back where attention weights are negligible.
**Updated assessment** (2026-03-27): Given turbo2's +8% uniform PPL hit, temporal decay would need to be very selective about which tokens to demote. May combine well with sparse V dequant (#51) — if attention weight is <1e-6, the requantization error doesn't matter.

### 28. turbo2 (2-bit) variant
**Status**: done — IMPLEMENTED AND TESTED
**Type**: new format
**What**: turbo2 = 2.5 bpv = 6.4x compression vs fp16. 2-bit PolarQuant (4-centroid Lloyd-Max), no QJL correction.
**Results** (2026-03-27, Qwen3.5-27B head_dim=256, 4K ctx, 4 chunks):
- turbo2 K+V: PPL 6.78 (+8.0% vs f16) — too much degradation for general use
- turbo3-K + turbo2-V: PPL 6.57 (+4.6%) — mixed is much better
- turbo2-K + turbo3-V: PPL 6.52 (+3.9%) — K matters more for quality
- turbo2-K + q8_0-V: PPL 6.49 (+3.4%) — best turbo2 combo
- Speed: identical to turbo3 (~30.6 tg128, compute-bound)
- KV memory: 20 MiB vs 28 MiB (turbo3) vs 128 MiB (f16) at 4K
**Conclusion**: turbo2 works but +8% PPL is too high for uniform K+V. Best use case is mixed turbo2-K/turbo3-V (3.9% gap at ~2.9 bpv average) or as part of layer-adaptive decay (#27). The 90% compression cliff warning from "Physics of KV Compression" paper was validated — turbo2 at 87.5% compression is indeed at the quality edge.
**Difficulty**: Medium (implemented in 1 session, 22 files modified).

### 29. Blackwell native FP4/FP6 tensor cores
**Status**: needs-research (hardware dependent)
**Type**: speed (future)
**Paper**: NVIDIA Blackwell `tcgen05.mma` with mixed FP4/FP6/FP8 inputs
**What**: On B200/RTX5090, turbo3's 3-bit values could be zero-padded to 4-bit and use native FP4 tensor cores. Eliminates dequant bottleneck entirely for Q*K matmul.
**Difficulty**: Medium (when targeting Blackwell). Long-term path.

### 30. Dynamic quantization switching at VRAM thresholds
**Status**: deferred — **complex, better solved by layer-adaptive modes**
**Type**: quality + memory
**What**: Auto-switch q8_0→turbo at VRAM pressure. Our layer-adaptive modes (#8-11) already provide the quality/compression tradeoff. Dynamic switching would add runtime complexity for marginal benefit over pre-configured layer-adaptive.

### 31. Turbo types in speculative decoding draft model
**Status**: done — **NO PRACTICAL BENEFIT**
**Type**: speed + memory
**What**: Draft models use `-ctkd`/`-ctvd` flags. turbo3 on draft KV saves VRAM.
**Results** (Qwen3.5-2B Q4_K_M draft → Qwen3.5-27B Q6_K target, n=256, draft=8):
  - q8_0 draft KV: 28.78 tok/s, n_drafted=1864
  - turbo3 draft KV: 28.85 tok/s, n_drafted=1936
  - Normal decode (no spec): ~31 tok/s
**Finding**: (1) Speculative decoding is slower than normal decode for this model pair — the 2B draft has poor acceptance rate. (2) turbo3 on draft KV has zero impact on throughput or acceptance because the 2B model's KV cache is negligible compared to the 27B target. turbo KV matters for the target model (which already uses it), not the draft.
**Conclusion**: turbo in speculative decoding is a non-issue. The draft KV is tiny and turbo3 doesn't affect acceptance rate.

### 32. Fused quantization in QKV projection
**Status**: deferred — **prefill already 98.8%, not worth the complexity**
**Type**: speed (prefill)
**Paper**: TurboAttention FlashQ (Microsoft, arXiv:2412.08585)
**What**: Fuse turbo quantization into the QKV projection pass rather than as a separate SET_ROWS step.
**Finding**: SET_ROWS is already negligible vs attention+FFN compute. Prefill at 98.8% of q8_0. Deep ggml graph integration not justified.

### 33. Entropy coding for stored/offloaded caches
**Status**: deferred — **storage-only, no quality/speed impact**
**Type**: compression (storage)
**Paper**: KVTC (NVIDIA, ICLR 2026, arXiv:2511.01815)
**What**: Arithmetic/Huffman coding on codebook indices for ~0.3-0.5 extra bits savings. Only helps disk storage/CPU offload. Revisit when prefix caching becomes a feature.

### 34. Cross-layer codebook sharing / delta coding
**Status**: deferred — **complex dependency, uncertain benefit**
**Type**: compression
**Paper**: XQuant (arXiv:2510.11236)
**What**: We already share a single codebook across all layers. Delta coding (encode layer N as delta from N-1) would add cross-layer dependency in the critical path — bad for pipeline parallelism. The benefit is extra compression, but we're already at 3.5 bpv which is sufficient.

### 35. HCAttention — values on CPU, keys on GPU
**Status**: deferred — **major arch change, future feature**
**Type**: memory (extreme context)
**Paper**: HCAttention (arXiv:2507.19823)
**What**: CPU-offloaded V with GPU-only K for extreme context (4M tokens). Good synergy with turbo-compressed K. Requires major architectural changes to attention pipeline.

### 37. MSE-optimal norm correction
**Status**: done — **NEGATIVE RESULT**
**Type**: quality
**What**: Replace L2-preserving norm correction (β = ||x||/||q||) with MSE-optimal scaling (α = ||x|| · dot(x,q) / ||q||²). Theoretically halves per-element MSE.
**Result**: turbo3 PPL 5.9083 vs baseline 5.8501 (+0.058, slightly WORSE). MSE-optimal reduces norm by cos(θ), lowering effective attention temperature. L2-preserving is better for attention.

### 38. Multi-model validation + KV cache context OOM fix
**Status**: done — **BUG FIX + QUALITY DATA**
**Type**: quality, bugfix
**Bug**: ggml context allocation for KV cache didn't account for turbo rotation matrix tensors (2 extra objects). Caused assertion failure on all non-Qwen3.5 models.
**Fix**: Add `n_turbo_extra` (4 tensors) to context size in `llama-kv-cache.cpp`.
**Quality results** (turbo3 uniform, 2K context):
  - Qwen3.5-27B Q6_K: +0.2% (excellent)
  - Qwen3.5-35B-A3B MoE Q4_K_S: +0.3% (excellent)
  - MN-Violet-Lotus-12B Q4_K_M: +2.6% (acceptable)
  - Qwen3-14B Q5_K_M: +3.8% (moderate degradation)
  - Gemma-3-27B-it Q4_K_M: turbo3 K+V +3.3%, turbo3-K +4.6% (V was broken pre-#45 fix)
**Key finding**: turbo3 quality degrades on head_dim=128 models (~3% PPL increase) vs head_dim=256 models (<0.3%). Gemma-3 V is completely broken due to SWA/global hybrid cache architecture. Needs investigation.

---

## New Experiments (from March 2026 research)

### 39. GSR Walsh ordering for FWHT
**Status**: done — **NEUTRAL** (no measurable improvement)
**Type**: quality (zero cost)
**Paper**: GSR (arXiv:2505.03810, ACL 2025 SRW)
**Branch**: `experiment/gsr-walsh-ordering`
**What**: Reorder FWHT output by sequency (sign-change count) using permutation `perm[s] = bit_reverse_7(gray(s))`. Groups similar-frequency components into the same turbo3 quantization block (32 elements within 128-element FWHT group).
**Results** (turbo3 uniform, 2K/8chunks):
  - turbo3 baseline: PPL 5.8323
  - turbo3 + Walsh ordering: PPL 5.8248 (-0.13%, within error bars ±0.164)
  - q8_0 reference: PPL 5.8375
**Finding**: GSR paper's massive gains (PPL 20.29→11.59) were without random sign arrays. Our PolarQuant rotation uses random signs (s1, s2) that already decorrelate all 128 output elements, making them identically distributed. Sequency reordering cannot improve intra-block variance when the signs already destroy frequency structure. Walsh ordering only helps fixed (non-randomized) Hadamard transforms.

### 40. Mean-centering before FWHT (HadaNorm)
**Status**: dropped — **duplicate of #22, incompatible with per-token quantization**
**Type**: quality (minimal cost)
**Paper**: HadaNorm (arXiv:2506.09932, Jun 2025)
**What**: Per-channel mean subtraction before Hadamard. Two interpretations: (a) per-token mean of 128-element group = what #22 tested (no benefit — already near-zero after L2 norm), (b) per-channel mean across tokens (requires calibration or running statistics, incompatible with per-token SET_ROWS pipeline). Same fundamental issue as #39: random sign arrays already decorrelate the distribution, making pre-centering redundant.

### 41. CAT alignment correction / per-channel scaling
**Status**: done — **PARTIALLY WRONG: InnerQ DOES help on hd128 (see #52)**
**Type**: quality
**Papers**: CAT (arXiv:2603.04359, Mar 2026), InnerQ (arXiv:2602.23200, Feb 2026)
**What**: CAT decomposes quantization error into concentration (outlier spread, what FWHT handles) and alignment (direction match, what FWHT does NOT handle). Alignment is INVARIANT to orthogonal rotations — FWHT cannot improve it.
**Analysis** (2026-03-27):
Our pipeline makes CAT-style interventions ineffective on head_dim=256:
1. **L2 normalization** removes per-channel magnitude → all vectors unit norm
2. **FWHT with random signs** mixes all channels → each output position ≈ i.i.d. N(0, 1/128)
3. **Per-channel scaling before FWHT**: destroyed by mixing (FWHT is a dense linear transform)
4. **Per-channel scaling after FWHT**: meaningless — all positions have identical distribution
5. **Lloyd-Max codebook** is already optimal for the resulting standardized Gaussian
6. **Norm correction** already preserves L2 norm of reconstruction (handles magnitude alignment)
**CORRECTION (2026-03-28)**: The analysis above is valid for head_dim=256, but on head_dim=128, per-channel scaling BEFORE L2 norm (i.e., before the pipeline above) DOES help. Channel magnitudes vary 20x on Qwen3-14B hd128 — normalizing channels before L2 norm changes the direction of the unit vector fed to FWHT, reducing quantization noise in the high-variance channels. This is what InnerQ (#52) implements. 55% gap closure on hd128.
**Closes research line**: #19, #20, #39, #40 still fail. #41 was partially right — per-channel scaling works when applied BEFORE L2 norm (changing the vector direction), not after (which is meaningless post-normalization).
**Difficulty**: Low (heuristic test) to Medium (full calibration).

### 42. KVLinC asymmetric K/V rotation strategy
**Status**: done — **NEGATIVE RESULT** (rotation helps keys too)
**Type**: quality
**Paper**: KVLinC (arXiv:2510.05373, Oct 2025)
**Branch**: `experiment/kvlinc-no-k-rotation`
**What**: KVLinC claims rotation helps V but hurts K. Test: disable K rotation (TURBO_NO_K_ROTATE=1), keep V rotation.
**Results** (turbo3 uniform, Qwen3.5-27B, 2K/8chunks):
  - turbo3 baseline (both rotated): PPL 5.8323 (-0.09% vs q8_0)
  - turbo3 K unrotated, V rotated: PPL 6.1647 (+5.6% vs q8_0)
  - turbo3 neither rotated (prior data): PPL 6.2357 (+6.8% vs q8_0)
**Finding**: Disabling K rotation hurts PPL by +0.33 (5.83→6.16). KVLinC's finding does NOT apply to our turbo3 codebook. Their result was for 2-bit with per-channel scale+zero quantization, which is a different paradigm. Our Lloyd-Max codebook with norm correction benefits from rotation on both K and V because it makes the distribution match the codebook's symmetric assumption. The +0.07 difference between K-unrotated (6.16) and neither-rotated (6.24) shows V rotation alone contributes about half the benefit.
**Risk**: Our norm correction assumes rotation — need to verify norm correction still works on unrotated K.
**Difficulty**: Medium (1 week). Straightforward to test.

### 43. SQuat-inspired query-orthogonal codebook selection
**Status**: needs-research
**Type**: quality
**Paper**: SQuat (arXiv:2503.24358, Mar 2025, Red Hat AI)
**What**: Instead of minimizing ||k - k_quantized|| (compression error), SQuat ensures quantization residual is **orthogonal to the query subspace**. Query subspace from prompt SVD (rank ~30 for d=4096), generalizes to response tokens. 2-bit, no fine-tuning, no calibration data.
**How it applies**: After FWHT rotation makes the distribution uniform, SQuat's orthogonality constraint ensures whatever residual quantization error remains is **invisible to queries**. The prompt-derived subspace can be computed once at prompt time with no ongoing cost. Could be combined with Lloyd-Max by biasing codebook selection toward codewords whose error is orthogonal to the query subspace.
**Challenge**: O(d³) for the iterative quantization algorithm. May be too expensive for per-token SET_ROWS. A simplified version (project quantization error onto query subspace and minimize that instead of MSE) could be cheaper.
**Difficulty**: High (2-3 weeks). Strongest theoretical backing for quality improvement.
**Research completed 2026-03-27**: See detailed analysis in memory/reference_squat_algorithm_deep_dive.md. Code at github.com/Red-Hat-AI-Innovation-Team/SQuat. Key finding: O(d^2) per-token cost is trivial at d=128. Critical tension: FWHT rotation may destroy the low-rank query subspace structure SQuat exploits. Recommended: test Option B (block iterative update) first, measuring both rotated and unrotated Q subspace rank.

### 44. PatternKV — pattern subtraction before codebook
**Status**: research-complete — **LOW expected value** (FWHT already solves distribution flattening)
**Type**: quality
**Paper**: PatternKV (arXiv:2510.05176, Oct 2025). Code: github.com/HCOOOH/PatternKV (0 stars, Llama-only Python/Triton).
**Research** (2026-03-27): PatternKV mines 32 centroids via KMeans during prefill, subtracts nearest pattern before INT quantization of residual. CRITICAL INSIGHT: FWHT and PatternKV are COMPETING solutions to the same problem (distribution flattening). After our sign-randomized FWHT, post-rotation distribution is already ~Gaussian — patterns would capture little residual structure. INT2 PatternKV gains +1pt LongBench over KIVI; our FWHT+Lloyd-Max already achieves near-q8_0 quality at 3-bit.
**Feasible path**: Subtract pattern before FWHT, store pre-rotated patterns for dequant. 4 patterns × 128 floats = 2KB/head. Pattern index costs 2 bits per 128-element group.
**V has a gotcha**: 75% of V vectors benefit; the rest must skip (quality collapses without adaptive threshold).
**Expected gain**: 0-0.5% on Qwen3.5-27B (already near-optimal), maybe 0.5-2% on head_dim=128 models.
**DeltaKV**: Dropped — requires trained MLP compressor, Sparse-vLLM, incompatible with our approach.
**Difficulty**: Medium. Could prototype but expected ROI is low.

### 45. Gemma-3 SWA V cache investigation
**Status**: done — **FIXED, Gemma-3 turbo3 now works**
**Type**: quality/bugfix
**Root cause**: V inverse rotation (`ggml_turbo_wht`) was missing from the iSWA `build_attn` overload in `llama-graph.cpp`. Gemma-3 uses iSWA for ALL layers.
**Fix**: Added V un-rotation block to iSWA `build_attn` overload at ~line 2235 (after `build_attn_mha`, before W_O).
**Results** (Gemma-3-27B-it Q4_K_M, 2K/8chunks):
  - q8_0: PPL 5.6995
  - turbo3 K+V: PPL 5.8867 (+3.3%) — **was 45 TRILLION before fix**
  - turbo3-K + q8_0-V: PPL 5.9633 (+4.6%) — was reported as +31% in earlier test
**Finding**: With the fix, Gemma-3 turbo3 quality matches the head_dim=128 model pattern (+3-4% PPL), same as MN-Violet-Lotus-12B (+2.6%) and Qwen3-14B (+3.8%). K-only slightly worse than K+V, consistent with Qwen3.5 findings that V matters more.

### 50. turbo4 K broken on head_dim=128 — missing Q pre-rotation for TURBO4_0
**Status**: done — **FIXED AND VERIFIED**
**Type**: bugfix
**What**: turbo4-K produced PPL 33K on Qwen3-14B (head_dim=128). turbo4-V worked fine.
**Root cause**: In `fattn.cu` line 702, `turbo_kv` only checked `GGML_TYPE_TURBO3_0`, NOT `TURBO4_0`. This gated Q pre-rotation at line 759. turbo4 K stored rotated, but Q never got pre-rotated → garbage dot products.
**Fix applied**: Changed Q pre-rotation guard at line 759 to include TURBO4_0:
```c
const bool turbo_k_any = (K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0);
if (turbo_k_any && Q->ne[0] % 128 == 0) {
```
**Results after fix**:
  - Qwen3.5-27B (head_dim=256): turbo4 K+V PPL 5.8186 (-0.32% vs q8_0) — **BEATS q8_0!**
  - Qwen3-14B (head_dim=128): turbo4 K+V PPL 6.9118 (+6.3% vs q8_0) — functional, turbo4-V still excellent (+1.9%)
  - Experiment #10's turbo4-K result (5.8451) confirmed reproducible after fix
**Mystery resolved**: Experiment #10 was probably from before FWHT rotation was implemented (no rotation = no pre-rotation needed).

### 51. Sparse V dequant (TheTom)
**Status**: done — **IMPLEMENTED AND VERIFIED**
**Type**: speed (decode)
**Credit**: TheTom (turboquant_plus/sparse-v-dequant)
**What**: Skip V dequantization for KV positions where `exp(score - max) < 1e-6`. At long context, 90%+ of attention weights are negligible.
**Implementation**: 3 lines added to fattn-vec.cuh V accumulation loop (both V_DOT2 and non-V_DOT2 paths). Threshold check after loading KQ_k, `continue` before V dequant.
**Results**: Zero quality loss (PPL bit-identical). On dense model, no speedup (attention <5% of compute). On MoE model, eliminates native dequant context scaling regression: 114.44→126.89 tok/s at 8K (+10.9%). Native dequant with sparse V now matches fp16 dequant speed at all contexts.
**Implication**: The fp16 decode dequant path may become unnecessary — sparse V achieves the same context-scaling fix with zero extra memory bandwidth.

### 52. InnerQ per-channel equalization (TODO-004)
**Status**: done — **~46% gap closure on head_dim=128, auto-disables on hd256**
**Type**: quality
**Papers**: InnerQ (arXiv:2602.23200), CAT (arXiv:2603.04359)
**Branch**: `experiment/innerq-channel-equalization`
**What**: Per-channel scaling before L2 norm + FWHT to equalize channel magnitudes. Inverse scaling on Q in FA kernel. Online calibration (100K token-counts), sqrt-dampened scales with clamp. Auto-detects balanced channels (max ratio < 1.2) and disables.
**Implementation**: Modified `turbo-quant-cuda.cuh` (device arrays, calibration with both RMS+max accumulators, K/V flag as kernel param, auto-detect), `fattn.cu` (inverse scaling in `k_turbo_fwht_forward`), `fattn-common.cuh` (inverse scale array), `set-rows.cu` (calibration state machine, K/V name detection).
**Key design decisions tested**:
  - K+V scaling beats K-only scaling (V direction adjustment helps even without output compensation)
  - Mixed K+V calibration stats beat K-only stats (V distribution contributes useful info)
  - Paper's max-based formula (`1/sqrt(max|K_i|)`) doesn't transfer to our codebook pipeline (PPL worse)
  - RMS-based with strength=0.20 is optimal for turbo3+FWHT+norm-correction
  - Auto-detect (max_ratio < 1.2 → disable) prevents harm on already-balanced hd256
**Results** (Qwen3-14B Q5_K_M, head_dim=128, 2K/8chunks):
  - turbo3 baseline: PPL 6.6340 (+3.33% vs q8_0 6.4206)
  - turbo3 + InnerQ (strength=0.20, RMS, K+V): PPL 6.5349 (+1.78%) — **46% gap closure**
  - K-only apply (strength=0.30): PPL 6.5418 (+1.89%) — 43% closure
  - K-only apply (strength=0.20): PPL 6.5477 (+1.98%) — 40% closure
  - Max-based mode=1 (paper's formula): PPL 6.6716 (+3.91%) — WORSE than baseline
**Results** (Qwen3.5-27B Q6_K, head_dim=256, 2K/8chunks):
  - turbo3 baseline: PPL 5.8501
  - turbo3 + InnerQ (forced): PPL 5.9283 — HURTS (+1.3%)
  - turbo3 + InnerQ (auto-detect): PPL 5.8501 — correctly disabled, no regression
**Env vars**: `TURBO_INNERQ=1`, `TURBO_INNERQ_STRENGTH=0.20` (default 0.5), `TURBO_INNERQ_MODE=0` (0=RMS, 1=max)
**Limitation**: Online calibration mismatch for first ~2K tokens (identity scales during calibration, final scales after). Run-to-run PPL variation of ~0.02 due to calibration timing noise.

### 46. BitDecoding-style dequant pipeline for turbo prefill
**Status**: needs-research → experiment #16b
**Type**: speed (prefill)
**Paper**: BitDecoding (HPCA 2026, arXiv:2503.18773), open source at github.com/OpenBitSys/BitDecoding
**What**: Register-level software pipeline: CUDA cores dequant tile N+1 while tensor cores MMA tile N. Uses `lop3` PTX for bit manipulation, `ldmatrix` for TC layout, XOR swizzling for bank-conflict-free shared memory. Drops dequant overhead from 40-50% to **15%**. GQA query reshape (relevant to our 24Q/4KV Qwen3.5 layout).
**Code available**: `csrc/bit_decode/` in the BitDecoding repo. C++ + CUDA, LibTorch build.
**How it applies**: Our turbo3 prefill (experiment #16) already achieves 98.8% of q8_0 via dequant-then-MMA. This would primarily benefit turbo4 prefill (currently stuck at 588 tok/s due to QJL fp16 precision loss). Inline dequant avoids the fp16 temp buffer entirely.
**Difficulty**: High (3-4 weeks). Restructure fattn-mma to add dequant pipeline stage.

### 47. ButterflyQuant — learnable O(n log n) transforms
**Status**: dropped — **NOT APPLICABLE** (weight quant only, random signs negate benefit)
**Type**: quality
**Paper**: ButterflyQuant (arXiv:2509.09679, Sep 2025, v3 Feb 2026)
**GitHub**: 42Shawn/Butterflyquant-llm (8 stars, README only, NO CODE)
**What**: Replace fixed Hadamard with learnable butterfly transforms parameterized by continuous Givens rotation angles. O(n log n) complexity, n*log2(n)/2 learnable params (448 for n=128). Each butterfly stage has 64 independent Givens rotations G(theta)=[[cos,-sin],[sin,cos]] at stride 2^(i-1). Uniformity regularization KL(P_bins||Uniform) for codebook utilization. Identity init (theta=0), SGD+cosine LR, 128 WikiText-2 samples, converges in 500-700 steps on single H100.
**Paper results**: W2A16 PPL 15.4 (ButterflyQuant) vs 16.43 (SpinQuant) vs 36.77 (GPTQ) on LLaMA-2-7B. Learned butterfly coherence 1.8-3.2e-2 per layer vs Hadamard's 1.56e-2.
**Why dropped**: Three fundamental blockers:
1. **Weight quant, not KV cache**: Paper optimizes rotation jointly with FIXED weight matrices. KV cache has dynamic per-token distributions — the learned rotation cannot be jointly optimized with the data it will see.
2. **Random signs already optimal**: Our S2*H*S1 pipeline achieves coherence ~1.56e-2 (Hadamard-optimal), BETTER than ButterflyQuant's learned 1.8-3.2e-2. Same reason GSR Walsh ordering (#39) showed zero benefit — random signs decorrelate all output elements, making any structured rotation improvement moot.
3. **No code, no KV validation**: GitHub repo is empty. Paper has zero KV cache experiments.
**CUDA cost if implemented**: 4 FMA per butterfly pair vs 2 add/sub for FWHT (2x compute per pass). Would need 896 twiddle factors (3.5KB) in constant memory per layer. Net: slower rotation for zero quality gain.
**Better alternative**: CAT alignment correction (#41) addresses the orthogonal "alignment" factor that our FWHT does NOT handle, rather than re-solving "concentration" that FWHT already handles optimally.

### 48. AQUA-KV — inter-layer KV prediction
**Status**: dropped — **architecturally incompatible with llama.cpp**
**Type**: quality + compression
**Paper**: AQUA-KV (arXiv:2501.19392, ICML 2025). Code: github.com/goodevening13/aquakv (19 stars, PyTorch only).
**Research** (2026-03-27): Impressive results (2-bit near-lossless) but three fundamental conflicts:
1. **Cross-layer dependency**: Requires dequanting layer L-1 KV to predict layer L. Our SET_ROWS quantizes each layer independently.
2. **Pre-RoPE requirement**: Must store pre-RoPE keys, re-apply on read. We store post-RoPE.
3. **Batch buffer model**: Accumulates 128 tokens then batch-compresses sequentially across layers. We quantize per-token.
Would require redesigning llama.cpp's entire KV cache write path. Not worth the architectural cost when our 3-bit approach already achieves near-q8_0 quality.

### 49. Tune parallel_blocks heuristic for turbo decode
**Status**: done — **NO EFFECT** (attention not the bottleneck)
**Type**: speed (decode)
**Branch**: `experiment/parallel-blocks-tuning`
**What**: Added GGML_PARALLEL_BLOCKS env var override to force different split-K values. Benchmarked all values from 1 to 32.
**Results** (turbo3 tg64 at 32K context, Qwen3.5-27B, RTX 3090):
  - default (auto): 29.95 tok/s
  - pb=1: 29.97, pb=2: 29.95, pb=4: 29.95, pb=8: 29.96, pb=16: 29.96, pb=32: 29.93
  - q8_0 baseline: 30.81 tok/s (turbo3 = 97.2%)
**Finding**: All parallel_blocks values within noise (±0.1 tok/s). Attention compute is <5% of total decode time — FFN dominates. The 2.8% turbo3-to-q8_0 gap is structural dequant overhead that can't be closed by attention-level tuning. This also applies to #17 (Split-K tuning).

### 50. Fix multi-sequence (n_seq > 1) turbo dequant
**Status**: done — **CRITICAL BUG FIX**
**Type**: correctness
**Branch**: `feature/turboquant-kv-cache`
**What**: Turbo dequant-to-fp16 kernels in fattn.cu ignored the stream dimension (ne[3]).
With kv_unified=false (the default) and n_seq > 1, K/V tensors have ne[3] = n_stream
during prefill. Only stream 0 was allocated and dequanted — streams 1+ read
uninitialized fp16 garbage, causing catastrophic PPL degradation.
**Fix**: Added ne[3]/nb[3] to kernel signatures, allocation sizes, and 3D grid launches
for all turbo dequant kernels (turbo3, turbo4) in both prefill and decode paths.
**Results**: n_seq=1: 6.31 (unchanged), n_seq=2: 6.30 (was 17.10), n_seq=4: 6.34 (was 22.56).

---

## External Research & References

### TheTom's validated findings (2026-03-26)
- **Layer-adaptive mode 2**: +0.37% PPL at 3.5x compression, strictly better than uniform turbo3
- **QJL stage unnecessary**: drop it, all bits to PolarQuant centroids, faster/simpler, PPL matched
- **fp16 centroid LUT**: decode +6-14% at long context, zero quality impact
- **Context-scaling fix (unrolled dequant byte extraction)**: flat 98.7-99.5% prefill through 32K
- **WHT/RoPE non-commutativity**: WHT must go after RoPE. Our code does this correctly (RoPE in model, FWHT in SET_ROWS/graph).

### TheTom's failed experiments
- **Custom GGML_OP_TURBO_WHT**: red herring, same speed as dense matmul
- **Group-32 rotation**: PPL 7.06 vs target 6.19 — full d=128 rotation necessary
- **Gemini's RoPE/WHT commutativity theory**: wasn't the actual issue

### TheTom's in-progress
- **M1 decode fix**: split 2×4-entry LUT for constant cache divergence (PPL identical, 4.4% M5 regression — investigating)
- **Hardware diagnostic script**: cross-platform benchmarking
- **Asymmetric K/V compression**: aligns with our experiment #10

### Ecosystem (as of 2026-03-27)
- **TheTom/turboquant_plus** (220+ stars, 91 commits, 511 tests) — Python reference, dropped QJL, 2747 tok/s prefill, 99% of q8_0 speed 2K-32K. Active: upstream llama.cpp PR prep, turbo4 fix, benchmark hardening.
- **TheTom/llama-cpp-turboquant** (34 stars, 11 forks) — Metal GPU, upstream for this repo. CUDA backend mentioned as in-progress but not yet validated.
- **tonbistudio/turboquant-pytorch** (338 stars, 42 forks) — Full PyTorch + Triton with QJL Stage 2. 3-bit: 99.45-99.61% cosine sim. MIT license.
- **Dejan.ai** — Fused Triton kernel for Gemma 3 4B on RTX 4090. 2-bit fused path: character-identical to fp16. 1.18-1.22x speedup.
- **0xSero/turboquant** — Triton + vLLM integration. 3-bit K, 2-bit V. Qwen3.5-27B on 4×RTX 3090: 914K token capacity (2x baseline), 30GB freed.
- **Aaryan-Kapoor** — CPU TQ3_0 in llama.cpp (block-32, 14 bytes/32 values, 3.5 bpw). Qwen3.5-35B: identical output to FP16 at temp 0.
- **veritatisquaesitoressumus** — CPU complete in ik_llama.cpp. TQ3 PPL 6.6872 vs FP16 6.5792. CUDA kernels written but unvalidated.
- **mudler** — Experimental branch with tq1_0/tq2_0/tbq3_0/tbq4_0 types. Issue #20977 (18 comments).
- **Madreag** — Ported Metal kernels to CUDA for RTX 5090: 4.6x KV compression, NIAH 6/6.
- **vLLM #38171** (39 upvotes) — Draft PR #38280 open with eval results. lishunyang12/vllm-omni PoC: NIAH 6/6, 7.5x cache reduction at 2-bit. CUDA/Triton kernels Phase 3.
- **Mainline llama.cpp** — Discussion #20969 (active), Issue #20977 (active). **No merged PR yet.** Maintainers want CONTRIBUTING.md compliance.
- **MLX** — Prince_Canuma implementation, Qwen3.5-35B 100% exact match 8.5K-64K. HuggingFace model available.
- **turboquant.net** — Community site. HN #1, 421 points, 119 comments.
- **Consensus across implementations**: Multiple devs independently dropped QJL (Algorithm 2), finding Algorithm 1 alone sufficient. (Our data contradicts this when norm correction is active — see #25.)

### Key papers
- TurboQuant (Google, ICLR 2026) — the original
- PolarQuant (Google, arXiv:2502.02617) — same authors, polar coordinate decomposition
- RotateKV (IJCAI 2025) — channel reordering + FWHT
- BitDecoding (HPCA 2026, arXiv:2503.18773) — TC-accelerated low-bit KV decode, **open source C++**
- SageAttention 1/2/3 (ICLR/ICML/NeurIPS 2025) — INT8/INT4 attention
- KVTuner (ICML 2025) — per-layer/head sensitivity analysis
- WUSH (arXiv:2512.00956) — optimal transform theory (impractical full, diagonal approx viable)
- NSNQuant (NeurIPS 2025) — calibration-free normalization (tested #22, no benefit)
- CommVQ (ICML 2025) — RoPE-commutative codebooks
- KVTC (NVIDIA, ICLR 2026) — 20x compression via transform coding
- Kitty (MLSys 2026) — uniform-precision tensor decomposition
- ~~SmoothRot (arXiv:2506.05413) — only targets FFN, not KV~~
- ConvRot (arXiv:2512.03673) — group Hadamard as convolution (dropped, TheTom tested)
- TurboAttention (Microsoft) — fused quant + FlashQ + SAS softmax (no code)
- HadaCore (arXiv:2412.08832) — TC-accelerated FWHT, 1.1-3.5x speedup
- AnTKV (arXiv:2506.19505) — attention-sink protection (tested #23, no effect)
- "More Keys Less Values" (arXiv:2502.15075) — asymmetric K/V theory

#### New papers (2026 survey, added 2026-03-27)
- **FlashAttention-4** (arXiv:2603.05451, Mar 2026) — Blackwell-optimized, 1613 TFLOPS/s B200, CuTe-DSL Python
- **SQuat** (arXiv:2503.24358, Mar 2025) — query-subspace orthogonal quantization error, 2-bit no calibration
- **CAT** (arXiv:2603.04359, Mar 2026) — concentration + alignment decomposition of quant error
- **KVLinC** (arXiv:2510.05373, Oct 2025) — asymmetric K/V: raw keys channel-wise, rotated values token-wise
- **DeltaKV** (arXiv:2602.08005, Feb 2026) — residual KV compression, 29% memory, Sparse-vLLM
- **BinaryAttention** (arXiv:2603.09582, Mar 2026) — 1-bit QK attention, 2x faster than FA2
- **Hadamard W_O** (arXiv:2603.08343, Mar 2026) — WHT replaces dense output projection, -25% params
- **GSR** (arXiv:2505.03810, ACL 2025) — Walsh (sequency) ordering for Hadamard, free PPL gain
- **HadaNorm** (arXiv:2506.09932, Jun 2025) — mean-centering before Hadamard
- **ButterflyQuant** (arXiv:2509.09679, Sep 2025) — learnable O(n log n) butterfly transforms
- **PatternKV** (arXiv:2510.05176, Oct 2025) — pattern subtraction before quantization
- **AQUA-KV** (arXiv:2501.19392, ICML 2025) — inter-layer KV prediction + residual quantization
- **MILLION** (arXiv:2504.03661) — product quantization for KV, codebook LUT in L1 cache
- **Physics of KV Compression** (arXiv:2603.01426, Mar 2026) — hallucination cliff at 90% compression
- **S2D** (arXiv:2602.14432, Feb 2026) — spectral origin of activation outliers
- **VQKV** (arXiv:2603.16435, Mar 2026) — training-free multi-codebook VQ for KV
- **ARKV** (arXiv:2603.08727, Mar 2026) — auto-select precision per layer via entropy/variance/kurtosis
- **KVzap** (arXiv:2601.07891, Jan 2026, NVIDIA) — learned importance prediction for KV pruning

---

## Dropped

### Group-32 rotation (ConvRot)
**Reason**: TheTom tested directly. PPL 7.06 vs target 6.19. Full d=128 FWHT rotation is necessary for proper decorrelation. Smaller group sizes lose too much.

### Custom GGML_OP_TURBO_WHT as speed optimization
**Reason**: TheTom found it's a red herring — same speed as dense matmul. Q pre-rotation moved inline into FA kernels (vec: shared memory FWHT, prefill: separate kernel with persistent buffer). V un-rotation stays at graph level for CUDA graph compatibility. Decode: 30.14 tok/s (-0.4% vs baseline), PPL identical. Key fix: `cudaMallocAsync` for Q temp buffer caused NaN on graph replay — replaced with persistent `cudaMalloc`.

### Gemini's RoPE/WHT commutativity theory
**Reason**: TheTom investigated, wasn't the actual root cause of quality issues. The real constraint is simpler: WHT must be applied after RoPE, which our implementation does correctly.

### SmoothRot (#20) — channel scaling before FWHT
**Reason**: Research (2026-03-27) found SmoothRot only targets FFN down-projection outliers, NOT KV cache or attention. Paper explicitly states smoothing before attention has "limited gains." Gains also vanish with GPTQ. Not applicable to our head_dim=128 KV cache quality gap.

### WUSH full transform (#21) — data-aware dense transform
**Reason**: O(d²) per block = 18x slower than FWHT for d=128. Requires per-model calibration + per-block matrix storage (64KB/block in fp16). Diagonal approximation is the only viable path — see CAT (#41) for a cleaner version of this idea.

### PatternKV (#44) — pattern subtraction before codebook quantization `needs-research`
**Paper**: arXiv:2510.05176 (Oct 2025, v2 Jan 2026). Online k-means (32 centroids/head), subtract nearest pattern, quantize residual only.
**Results**: +1pt LongBench at INT2 over KIVI, +2.6pt GSM8K. Only 0.08% drop at INT4. +6.6% prefill latency, +2.6% decode latency.
**Analysis (2026-03-27)**: LOW expected benefit for turbo3/turbo4 because FWHT rotation already flattens the distribution (exactly what PatternKV targets). PatternKV operates on raw asymmetric INT quant without any rotation, so the "peaked distribution with outliers" problem it solves is already handled by our pipeline. Pattern subtraction in rotated space would require rotating patterns too, and rotated-space values are already approximately Gaussian where Lloyd-Max codebook is near-optimal. The V cache adaptive threshold (75% utilization, catastrophic without it) adds fragile complexity. Would need to operate pre-FWHT on raw vectors, but then pattern centroids are head_dim=128/256 floats that must be stored per-head per-layer (32 * 128 * 4B = 16KB/head, 40 layers * 4 heads = 2.56MB for Qwen3.5-27B -- fits in constant memory). **Verdict: unlikely to beat our existing FWHT flattening, but a simplified 4-pattern version is low-risk to prototype.**

### 53. Inverse-FWHT prefill dequant (from dusterbloom TBQ PR#2) `done`
**Source**: PR #2 (dusterbloom). TBQ prefill does full inverse Hadamard + sign flip during dequant-to-fp16, producing original-domain fp16 values. Standard MMA then works with no Q rotation.
**Why**: Our turbo4 prefill was 37% of q8_0 because we bypassed MMA to avoid fp16 centroid precision loss. TBQ's approach avoids the problem — the inverse FWHT mixes centroid values in float32 shared memory, only casting to fp16 after the transform.
**Result**: turbo4 prefill 420 → **1124 tok/s** (+167%, matching q8_0). PPL 5.858 (+0.46% vs vec, within noise). Decode unchanged at 30.17 tok/s.
**Design**: K-only inverse FWHT (V uses simple dequant, fp16 loss negligible). Q NOT pre-rotated (K in original domain). InnerQ inverse scaling applied in kernel. New kernel `k_turbo4_dequant_f16_inv_fwht` — 128 threads, shmem butterfly.

### 54. Binary search quantization with pre-computed boundaries `done`
**Source**: PR #2 (dusterbloom). Uses pre-computed Lloyd-Max decision boundaries with O(log n) binary search tree instead of linear codebook scan.
**Result**: Already implemented — `turbo_find_nearest_4bit()` with `d_turbo_mid_4bit[15]` does 4 comparisons for 16 centroids. Nothing to do.

### 55. N(0,1) centroid normalization (unnormalized Hadamard) `dropped`
**Source**: PR #2 (dusterbloom). Uses unnormalized Hadamard so post-FWHT values are N(0,1). Centroids universal for any head_dim.
**Result**: Functionally equivalent to our approach — the 1/√128 factor just moves between centroids and norm. No quality/speed benefit, breaking format change. Not worth it.

### 56. Norm correction — adjust stored norm for centroid reconstruction error `done`
**Source**: PR #5 (dusterbloom). After Lloyd-Max quantization, recompute stored norm as `||x_original|| / ||x_reconstructed_centroids||` so dequantized vector has correct L2 norm.
**Why**: Our current norm is just `||x||`. After quantizing to centroids, the reconstructed unit vector's L2 norm isn't exactly 1.0 — it depends on which centroids were selected. Multiplying by the original norm gives the wrong magnitude. The correction is: `norm_corrected = ||x|| / ||centroid_recon_unit||`. Zero runtime cost (correction during quant in SET_ROWS, dequant path unchanged).
**Risk**: Very low — format-compatible (still stores one fp16 norm per block), dequant kernels unchanged.
**Expected gain**: Small PPL improvement, especially at lower bit rates (turbo2/turbo3) where centroid quantization error is larger.

### 57. Persistent decode K/V fp16 buffers `done`
**Source**: PR #5 (dusterbloom). Replace `cudaMallocAsync`/`cudaFreeAsync` per decode token with grow-only persistent `cudaMalloc` buffers (same pattern as our existing `q_rot_buf`).
**Why**: Avoids async allocator overhead on every decode step. Already proven pattern in our Q rotation buffer.
**Risk**: Very low — same pattern as `q_rot_buf`.
**Expected gain**: Small decode latency reduction, mainly visible at short context where allocator overhead is proportionally larger.

### 58. Head dimension padding to 128 `done`
**Source**: PR #5 (dusterbloom). Pad KV cache dimensions to nearest 128 boundary for models with non-128-aligned head_dim.
**Implementation** (branch experiment/tbq-ideas): Zero-pad per-head K/V dimensions. Padding in llama-kv-cache.cpp (allocation, stream views, get_k/v, cpy_k/v) and llama-graph.cpp (Q padding + output crop in build_attn, both normal and iSWA).
**Test results** (Phi-3-mini, head_dim=96→128): turbo3 +3.97%, turbo2 +22.5%. No regression on 128/256-aligned models (Qwen3.5-27B exact match).
**Design**: Padding is per-head to prevent FWHT groups crossing head boundaries. Zero-padded dimensions are no-ops in dot product by Parseval's theorem.

### 59. Smooth scaling before FWHT (from VecInfer) `done`
**Paper**: arXiv:2510.06175 (Oct 2025). VecInfer applies SmoothQuant-style per-channel scaling before Hadamard rotation.
**Why**: FWHT redistributes outliers but doesn't fix inter-channel magnitude imbalance. Smooth scaling divides each K channel by `λ_i = sqrt(max(|K_i|))` (calibrated offline), then multiplies Q by the same factor. VecInfer ablation shows smooth + Hadamard is super-additive: Hadamard alone 51.0, smooth+Hadamard 51.8 at 1.5-bit on LLaMA-3.1-8B. Similar to our InnerQ but targets a different axis (magnitude equalization vs variance equalization).
**Implementation**: Calibrate per-layer channel scales from ~256 samples. Apply inverse scale to K in SET_ROWS before FWHT. Apply forward scale to Q before attention. Store scales in constant memory (128 floats/layer). Could reuse InnerQ infrastructure.
**Risk**: Low — same pattern as InnerQ. Calibration required.
**Expected gain**: Small PPL improvement at low bit rates (turbo2/turbo3). Negligible at turbo4 where we're already beating q8_0.
**Result**: Already covered by InnerQ (#52). VecInfer's formula = InnerQ mode=1 (max-based), which was tested and found WORSE than baseline on turbo3. InnerQ mode=0 (RMS, strength=0.20) gives 46% gap closure on hd128, auto-disables on hd256 (already balanced). No additional work needed.

### 60. Product quantization + ADC lookup tables for K (from VecInfer) `needs-research`
**Paper**: arXiv:2510.06175 (Oct 2025). VecInfer uses Product Quantization (PQ) instead of scalar VQ for keys. The key insight: PQ enables Asymmetric Distance Computation (ADC) — precompute `query_subvec · centroid` lookup table once per query, then each key token's attention score is just M integer-indexed table lookups instead of full dequant + dot product. For d=128, M=64 sub-vectors of dim 2: 64 uint8 lookups + 63 adds vs 128 dequant + 128 FMA.
**Why**: Fundamentally faster Q·K computation. VecInfer reports 2.7x attention speedup on H100 at 2-bit. The fused kernel (compute on quantized, never dequant K) is the dream path.
**Implementation**: Major arch change — replace scalar Lloyd-Max codebook with PQ codebook (M sub-vectors × C centroids). Requires offline codebook training (faiss k-means, ~256 calibration samples). New SET_ROWS kernel (PQ encode), new FA kernel (ADC LUT + table lookup). Format change.
**Risk**: High — complete rework of quantization + attention for K. V can stay scalar. Python prototype first to validate quality (PQ MSE vs scalar Lloyd-Max MSE at same bit rate).
**Expected gain**: Potentially large decode speedup at long context. Quality may differ — PQ sub-vector independence assumption vs scalar per-element optimality.

### 61. Trellis-coded quantization (TCQ) for turbo2 + turbo3 `tested-success`
**Source**: QTIP (NeurIPS 2024, arXiv:2406.11235) + Marcellin & Fischer 1990. From V.34 modem TCM, adapted for source coding. Cross-field idea from telecoms.
**Concept**: Replace independent Lloyd-Max scalar quantization with Viterbi-optimal joint quantization over 128 elements using a bitshift trellis. Each element still stores k bits, but a trellis with 2^L states constrains which codewords can follow each other, enabling a much larger effective codebook (2^L entries vs 2^k) at the same bit rate.
**Key insight — bitshift trellis decode is O(1)/element**: Element t's reconstruction depends only on an L-bit sliding window at bit positions [t*k, t*k+L). Fully parallel GPU decode, ~3-5 instructions per element. Codebook fits in shared memory.
**Python prototype results** (i.i.d. N(0,1), 128 elements, trained codebook via GLA):
- 2-bit L=8 (256 states): MSE 0.110 (+6.2% vs Lloyd-Max, +0.28 dB)
- 3-bit L=6 (64 states): MSE 0.035 (+20.3% vs Lloyd-Max, +0.99 dB)
- **3-bit L=9 (512 states): MSE 0.031 (+30.3% vs Lloyd-Max, +1.57 dB)**
- Codebook training essential — untrained codebooks show zero or negative gain
- V=1 (scalar) trellis, not QTIP's V=2 (vector). QTIP reports larger gains with V=2.
- D(R) bound still 4.54 dB away at 3-bit — room for improvement with larger trellis/V=2
**Encode**: Viterbi algorithm, O(2^L × 128) per block. At L=9 (512 states): ~6.4 ms/token for 320 blocks.
**Next steps**:
1. CUDA encode kernel: Viterbi in shared memory, warp-parallel state updates
2. CUDA decode kernel: bitshift window + codebook lookup in fattn-vec and fattn MMA paths
3. New block format: same qs[] layout but interpretation changes (trellis indices, not independent)
4. Codebook training: offline (calibration data), store per-model trained codebook
**Risk**: High engineering effort. Encode is inherently sequential over 128 elements (Viterbi). No published KV cache TCQ exists — would be novel.
**Priority**: turbo3 (30% MSE reduction at L=9 is large). turbo2 gains more modest (6% at L=8), needs larger trellis.
**Prototype**: `scripts/tcq_prototype.py`
**CUDA Implementation Results** (2026-03-28):
- turbo3_tcq (3.25 bpv): PPL 5.8294 (-0.14% vs q8_0), prefill 894 tok/s, decode 28.69 tok/s
- **turbo2_tcq (2.25 bpv): PPL 6.0546 (+3.7% vs q8_0, -61% vs turbo2's 15.61)**, prefill 976 tok/s, decode 29.53 tok/s
- Free-init optimization: allow any initial trellis state (all 2^L states equally viable). 37.6% MSE gain at 3-bit.
- Key bugs fixed: backtrace race condition (4→2 per byte packing), normalization order (must apply InnerQ BEFORE normalize)
- **turbo2_tcq is the publication story**: at 2.25 bpv, gets near-3-bit quality (6.05 vs 5.83). First TCQ for KV cache.
**Optimized Codebooks** (2026-03-29, numpy GLA n_train=4000, 100 iters):
- 3-bit: 37.6%→50.1% MSE reduction. turbo3_tcq PPL 5.8331 at 2K, **+0.81% at 32K** (vs turbo3's +1.80%)
- 2-bit: 4.2%→33.1% MSE reduction. turbo2_tcq PPL 6.0592 at 2K (was 6.1826 with old codebook)
- TCQ halves turbo3 context degradation: +1.04% at 65K vs turbo3's +2.28%
- TCQ transforms turbo2 long-context: +4.20% at 32K vs turbo2's +8.58%
- Best hybrid: turbo2_tcq-K + turbo3_tcq-V (2.75 bpv): +1.44% at 2K, +2.99% at 65K
- CUDA-trained codebooks find lower MSE but worse PPL — different GLA local optima. numpy codebook is best.
- Training scripts: `scripts/tcq_train_vectorized.py` (best), `scripts/tcq_train_cuda.cu`

### 62. Per-32 norm within 128-element rotation groups `planned`
**Source**: signalnine/llama-cpp-turboquant (NVIDIA contributor). Merged into TheTom's tree 2026-03-29.
**Concept**: FWHT over 128 elements, but store a separate corrected norm per 32-element sub-block. 4x finer norm granularity captures local amplitude variation in post-WHT data.
**Tradeoff**: +0.5 bpv overhead. At 2-bit: 2.5 bpv. turbo2_tcq at 2.25 bpv gets 6.06, signalnine's per-32-norm at 2.5 bpv gets 6.01. Similar quality, different approach.

### 63. Parallel FWHT encode (all-thread butterfly) `done`
**Source**: signalnine's set-rows.cu. 128 threads participate in FWHT via shared memory butterfly.
**Result**: Implemented on experiment/tbq-ideas branch. Verified identical output to serial kernel. Expected 1.5-2x encode speedup at long context.

### 64. 64-element WHT group fallback + zero-padding `done`
**Source**: signalnine. Pad head_dim to next multiple of 128.
**Result**: Implemented as experiment #58 (zero-padding approach). Always GROUP_SIZE=128, no template branching needed.

### 65. Sparse V dequant — skip negligible attention weights `planned`
**Source**: signalnine/TheTom. Skip V dequant+accumulation when all attention weights < 1e-6.
**Expected gain**: +22.8% decode at 32K (TheTom's number). More at longer context. Zero quality impact.
**Risk**: Very low — ~3 lines in fattn-vec.cuh, purely additive.

### 66. Fused attention+dequant kernel — eliminate fp16 intermediate `planned`
**Concept**: Dequant K/V inline during attention instead of pre-dequanting to fp16 buffer. For turbo3: 3 instructions per element (extract index, load centroid, multiply norm). For TCQ: sliding window + codebook lookup.
**Expected gain**: 1.5-2x decode speedup at long context. This is the fork-justifying feature — nobody else has fused attention for custom VQ codebooks.
**Risk**: High — requires rewriting fattn-vec and fattn-mma kernels. Register pressure is the main concern.

### 67. Speculative decoding with turbo-enabled larger draft models `planned`
**Concept**: turbo KV on target model frees VRAM, allowing a larger/better draft model. Qwen3.5-27B Q6_K + turbo3 KV at 64K frees ~2.2 GB vs q8_0 — enough for 3B draft in Q4 instead of 0.5B.

### 68. 256-element TCQ blocks for head_dim=256 models `planned`
**Concept**: Run Viterbi trellis over 256 elements (full head_dim) instead of two independent 128-element groups. Longer trellis = better coding gain. Only benefits head_dim=256 models.
**Priority**: Medium — nice paper result showing TCQ scales with block length.

### 69. Temperature scaling — attention sharpening via norm inflation `done`
**Source**: Competitive analysis 2026-03-31. Duster's TBQ accidentally inflates norms 2.77x, acting as attention temperature T=0.36.
**Result**: **MASSIVE WIN.** Alpha=1.20 optimal for both 3-bit and 2-bit TCQ. 5-14% PPL improvement at ALL context lengths. No regression anywhere. We now beat every competitor at every context length at both bit rates.
**Implementation**: `d_tcq_norm_alpha` constant in turbo-quant-cuda.cuh, loaded from `TURBO_TCQ_ALPHA` env var in set-rows.cu. Applied to both 3-bit and 2-bit encode kernels.
**Key numbers**: 3-bit @64K: 6.224 (was 7.034, TBQ3 was 7.034). 2-bit @64K: 6.248 (was 7.222, TBQ2 was 7.332).
**Default**: Hard-code alpha=1.20 for shipping. Keep env var for experimentation.

### 70. Asymmetric K/V norm scaling `done`
**Source**: Competitive analysis quality findings. K temperature helps attention routing, V accuracy helps output quality.
**Concept**: Separate alpha_k and alpha_v via `TURBO_TCQ_ALPHA` and `TURBO_TCQ_ALPHA_V` env vars.
**Change**: Added `d_tcq_norm_alpha_v` constant, K/V-conditional alpha in both 3-bit and 2-bit encode kernels.
**Result**: **V-heavier asymmetric (αK=1.10, αV=1.30) beats symmetric α=1.20 by 0.06-0.10 PPL at 8K-64K** with no 2K regression. Key finding: **V scaling contributes 6.5x more to quality than K scaling**, challenging the "attention temperature" narrative. The benefit comes from V magnitude restoration, not attention routing sharpness. Optimal at 64K: αK=1.05, αV=1.35 (−0.107 PPL vs symmetric).

### 71. Native `vec_dot_fattn_vec_KQ_turbo3_tcq` — inline TCQ decode in FA `rejected`
**Source**: Competitive analysis speed gap. Duster has `vec_dot_fattn_vec_KQ_tbq3_0` for TBQ. We dequant all KV to f16 first.
**Concept**: Read 9-bit state from bitstream → `codebook[state] * norm` → accumulate dot product inline in FA. No intermediate f16 buffer needed.
**Change**: New function in fattn-common.cuh. Simpler than TBQ's vec_dot (no inverse Hadamard).
**Expected**: ~7% decode speedup, reduced VRAM at long context (eliminates O(context) f16 buffer).
**Result**: **0% improvement** on both dense Qwen3.5-27B and MoE Qwen3.5-35B-A3B. The dequant-to-f16 path is already fast enough that eliminating the intermediate buffer doesn't help — the bottleneck is elsewhere.

### 72. Chunked cuBLAS GEMM prefill `rejected`
**Source**: Competitive analysis speed gap. Duster's implementation: 3-kernel pipeline (init, softmax-update, finalize) + `cublasGemmStridedBatchedEx`.
**Concept**: Dequant 4096 KV tokens at a time to f16, use cuBLAS for Q@K^T and P@V with online softmax between chunks. NOT TCQ-specific — works for all quant types.
**Change**: New prefill path in fattn.cu. Reference: Duster's fattn.cu lines 502-1005.
**Expected**: 20-27% prefill speedup. Enables 350K+ context on single RTX 3090.
**Result**: **1-5% SLOWER than fused MMA.** pp512: -1.3%, pp2048: -2.6%, pp4096: -3.3%, pp8192: -5.0%. The fused MMA flash attention is fundamentally better because it avoids materializing the O(nq×nkv) score matrix S. cuBLAS GEMM must write/read S as intermediate, adding significant memory bandwidth overhead that dominates any tensor core advantage. Also found a bug in Duster's code: `CUBLAS_OP_N` with `lda=D < M=chunk_len` (undefined behavior). Not worth pursuing further — the architecture is fundamentally disadvantaged vs fused kernels on modern GPUs.

### 73. Parallelize TCQ encode thread-0 serial sections `done`
**Source**: Competitive analysis encode speed. Thread-0 does FWHT rotation + backtracking + bitpacking alone.
**Concept**: FWHT is 128 elements × 7 stages — use 64+ threads (butterfly pattern, already done for #63 but only for non-TCQ). Bitpacking: each thread packs its own segment.
**Change**: turbo-quant-cuda.cuh TCQ encode kernel.
**Result**: **+12.6% prefill speedup** (3-bit: 902→1019, 2-bit: 983→1102 t/s at pp512). Warp-level reductions added in simplify pass. Merged to master.

### 74. TCQ error decorrelation via element permutation `dropped`
**Source**: Competitive analysis quality findings. TCQ trellis (right-shift, k=3) shares 6/9 state bits between consecutive positions → correlated errors. Autocorrelation ~0.15-0.30 at lag 1. Correlated errors average out slower in Q@K dot products.
**Concept**: Apply fixed permutation (e.g., bit-reversal) to element indices after FWHT before trellis encoding. Decorrelates errors across d_k dimension without changing MSE. Matching inverse permutation in decode.
**Risk**: Medium — needs careful verification that permuted trellis still converges. May interact with codebook optimality.
**Test**: Measure lag-1 autocorrelation before/after, PPL at 2K-64K.

### 75. Lloyd-Max boundaries as TCQ initial state prior `needs-research`
**Source**: Duster's TBQ uses textbook-optimal N(0,1) Lloyd-Max centroids. These are MSE-optimal for scalar Gaussian quantization.
**Concept**: Use Lloyd-Max bin boundaries to inform TCQ Viterbi initial state distribution or trellis path metric initialization. The optimal scalar quantizer boundaries partition the space in a way that might help Viterbi converge faster or to better paths.
**Risk**: Speculative — trellis structure already constrains state transitions heavily.

### 76. Optimal temperature grid search across context lengths `done`
**Source**: Follow-up to #69/#70. Does optimal αV vary by bit-rate and context length?
**Concept**: Sweep αV from 1.0 to 1.6 (αK=1.1 fixed) for turbo3_tcq and turbo2_tcq at 2K/8K/32K.
**Result**: αV=1.3 is robustly optimal for 3-bit at all contexts. 2-bit has slight preference for αV=1.35 at 8K (−0.052 PPL) but regresses at 2K (+0.041). Difference too small for per-bit-rate defaults. **Current defaults confirmed optimal. No changes.**

### 77. Verify turbo4 quality gap vs TBQ4 `done`
**Source**: Competitive analysis: TBQ4 beats turbo4 by 0.01-0.03 PPL everywhere. turbo4 has no TCQ — just scalar 4-bit quantization.
**Concept**: Temperature scaling (hypothesis b) was the answer. Added `d_turbo4_alpha` constant with `TURBO4_ALPHA` env var override.
**Result**: α=1.2 optimal (same as symmetric TCQ). PPL improvement: −0.236 (2K), −0.637 (8K), −0.482 (32K). turbo4 now BEATS q8_0 at all context lengths. V matters more than K (same as TCQ). Hardcoded α=1.2 as default.

### 78. Measure TCQ error autocorrelation empirically `done`
**Source**: Finding #29 from competitive analysis. Theoretical prediction of lag-1 autocorrelation ~0.15-0.30 but never measured.
**Concept**: Dumped post-FWHT values and output symbols from kernel. Computed errors and autocorrelation in Python.
**Result**: Theoretical prediction was WRONG. Both 3-bit and 2-bit TCQ errors are effectively iid (lag-1 autocorrelation ≈ -0.008, matching iid baseline exactly). FWHT rotation destroys trellis error structure. Experiment #74 can be dropped.

### 79. TBQ-style encode as TCQ fallback for speed-critical path `needs-research`
**Source**: Duster's TBQ encode is fully parallel (one binary search per element, ~660B shared mem) vs our Viterbi (128 sequential barrier-synced iterations, 34.5KB shared mem).
**Concept**: Offer Lloyd-Max scalar quantize as a fast encode path (e.g., for streaming/real-time), with TCQ Viterbi as the quality path. Runtime flag to select. Zero code sharing issues — the two encoders write the same bitstream format if we match centroids.
**Risk**: Quality regression vs TCQ. Only useful if encode speed matters (batch inference, very long prompts).

### 80. Padding non-128 head_dim completion `ready`
**Source**: Infrastructure comparison — Duster has it done, ours is "in progress" (#58/#64).
**Status**: Code changes done, needs final build+test verification.
**Test**: Verify PPL on a head_dim=128 model (e.g., Llama-3) and head_dim=256 model (Qwen3.5).

### 81. Sparse V dequant integration with TCQ `done`
**Source**: #65 planned but not tested with TCQ path. TheTom showed +22.8% decode at 32K.
**Concept**: Skip V dequant+accumulation when all attention weights in a block < threshold. Works with both f16-dequant path and future native vec_dot (#71).
**Change**: ~3 lines in fattn-vec.cuh. Purely additive.
**Result**: Already implemented in commit 78d6bb5a0. Active for all quant types in fattn-vec decode path. Not yet benchmarked independently for TCQ.

### 82. Replicate TBQ's exact 1-bit behavior for validation `needs-research`
**Source**: Bombshell finding — TBQ is accidentally 1-bit. 100% of post-FWHT values map to 2 inner bins.
**Concept**: Implement explicit 1-bit quantization (just store sign + raw norm) and verify it matches TBQ3 PPL exactly. If it does, this conclusively proves the temperature theory and means TBQ's 3-bit storage wastes 2 bits.
**Test**: PPL should match TBQ3 at all context lengths. If so, we can publish this finding.

### 83. Adaptive temperature per layer `needs-research`
**Source**: From MSE-PPL divergence investigation — Q anisotropy varies wildly per layer (layer 19: rank 1.9, layer 20: rank 114). Optimal attention temperature likely varies per layer too.
**Concept**: Store per-layer alpha in constant memory. Use different temperature scaling per layer based on that layer's Q effective rank.
**Risk**: Requires knowing Q statistics at quantization time (not available in llama.cpp's online KV quantize). Could use precomputed per-model tables.

### 84. 350K+ context validation `blocked`
**Source**: Duster's chunked cuBLAS GEMM enables 350K+ on single 3090. Currently we OOM much earlier.
**Concept**: After implementing #72, benchmark at 128K/256K/350K. Verify PPL doesn't degrade, measure speed.
**Depends**: #72 (chunked cuBLAS GEMM prefill) — **REJECTED** (slower). Need alternative approach for long-context VRAM savings.

### DeltaKV (#44b) — inter-token residual compression `dropped`
**Paper**: arXiv:2602.08005 (Feb 2026). Learned MLP compressor, strided reference tokens, global L2 retrieval.
**Analysis**: Requires training (~8 GPU hours per model), learned projections (MLP weights per layer), and a full framework rewrite (Sparse-vLLM). Fundamentally incompatible with our fixed-codebook approach. The per-token reference lookup is O(S) per token, not feasible in a CUDA kernel during SET_ROWS. **Verdict: wrong paradigm for llama.cpp integration.**
