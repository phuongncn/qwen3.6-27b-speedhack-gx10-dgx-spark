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
**Status**: done — **turbo3 only** (turbo4 disabled due to QJL fp16 precision loss)
**Type**: speed (prefill)
**Branch**: `experiment/prefill-dequant-attend`
**Result**: turbo3 prefill 631→1125 tok/s (1.78x, 98.8% of q8_0). PPL 5.8501 (within error bars of 5.8323 baseline).
**turbo4 issue**: QJL correction (~0.001 magnitude) rounds away in fp16 temp buffer. turbo4 prefill PPL 5.8966 vs 5.8186 vec kernel (+1.3%). Disabled — turbo4 uses vec kernel for prefill (588 tok/s).
**Bug fixed**: turbo4 dequant_f16 kernel had missing block indexing for ne0 > QK_TURBO4 (Qwen3.5-27B has head_dim=256).

### 16b. turbo4 prefill via float32 temp or inline MMA dequant
**Status**: ready
**Type**: speed (prefill, turbo4 only)
**What**: Two options to fix turbo4 prefill: (a) Use float32 temp buffer instead of fp16 (2x memory, need MMA dispatch that reads float32 K/V), or (b) BitDecoding-style inline dequant in MMA pipeline (no temp buffer, dequant overlapped with TC matmul). Option (b) is the long-term solution.
**Difficulty**: Medium (a) to High (b).

---

## Needs Research — Prefill Speed (mostly solved, 98.8% of q8_0)

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

### 16. Prefill-specific dequant-then-attend
**Status**: needs-research
**Type**: speed (prefill)
**What**: Separate prefill path: batch-dequantize KV to fp16 temporary buffer, run standard fp16 MMA attention on it. Trades temporary VRAM for q8_0-equivalent prefill speed.
**Challenge**: Memory overhead. 32K context × 128 head dim × 2 bytes × num_heads per layer. But turbo's whole point is saving KV VRAM — spending some temporarily during prefill is a reasonable tradeoff.
**Difficulty**: Medium (1-2 weeks). Simplest approach to prefill parity.

---

## Needs Research — Decode Speed (polish: 95-97% → parity)

### 17. Split-K / FlashDecoding for very long context
**Status**: needs-research
**Type**: speed (decode)
**Paper**: FlashDecoding (Stanford), FlashDecoding++ (MLSys 2024)
**What**: Split KV sequence across multiple thread blocks, reduce results. FlashDecoding++ adds async softmax eliminating 18.8% sync overhead. Vulkan backend already has `flash_attn_split_k_reduce.comp`.
**How it applies**: At very long context (64K+), single thread block can't process KV fast enough. Split-K would help. Also, async softmax could close the remaining 3-5% decode gap.
**Difficulty**: Low-Medium (1-2 weeks).

### 18. SAS softmax optimization
**Status**: needs-research
**Type**: speed (both)
**Paper**: TurboAttention (Microsoft, arXiv:2412.08585)
**What**: Decompose `exp(-x) = LUT(-x_int) * polynomial(-x_dec)`, polynomial runs on tensor cores in FP16. Independent of KV quant type.
**Difficulty**: Medium (1 week). 5-15% improvement, orthogonal to everything else.

---

## Needs Research — Quality Improvements

### 19. Channel reordering before FWHT
**Status**: needs-research
**Type**: quality (zero decode cost)
**Paper**: RotateKV (IJCAI 2025, arXiv:2501.16383)
**What**: Sort channels by outlier magnitude before applying Hadamard transform. Adapts to varying channel-wise outlier distributions without losing FWHT efficiency. Also applies pre-RoPE grouped-head rotation to smooth outliers across heads.
**Result in paper**: <0.3 PPL degradation at 2-bit on LLaMA-2-13B, 3.97x memory reduction.
**How it applies**: Add a learned permutation vector (one per model, computed during calibration) that reorders channels before FWHT in SET_ROWS. Inverse permutation after dequant. The permutation is just an index lookup — essentially free.
**Difficulty**: Low (a few days). High potential quality win.

### 20. SmoothRot — channel-wise scaling before FWHT
**Status**: needs-research
**Type**: quality (minimal decode cost)
**Paper**: SmoothRot (arXiv:2506.05413, Jul 2025)
**What**: Per-channel scaling factors applied before Hadamard rotation. Reduces outlier magnitude so FWHT produces more uniform post-rotation distribution. 10-30% quantization gap reduction, no added latency at decode.
**How it applies**: Store per-channel scale factors (computed from calibration data). Multiply before FWHT in SET_ROWS, divide after dequant. The divide can be folded into the norm.
**Difficulty**: Low-Medium (1 week). Complements channel reordering (#19).

### 21. WUSH — data-aware transform replacing pure FWHT
**Status**: needs-research
**Type**: quality
**Paper**: WUSH (arXiv:2512.00956, Nov 2025, ISTA/ETH)
**What**: Proves Hadamard is the optimal *data-agnostic* orthogonal transform. Then derives a non-orthogonal transform: Hadamard + data-dependent SVD-based diagonal scaling. Reduces error by up to d× (d=block size). Consistently beats pure Hadamard on MXFP4 and INT4.
**How it applies**: Replace FWHT with WUSH in SET_ROWS. The diagonal scaling factors would need to be stored per-model or per-layer (small overhead). Decode dequant would need the inverse scaling — can fold into norm.
**Difficulty**: Medium (1-2 weeks). Strongest theoretical backing.

### 22. NSN normalization for universal codebooks
**Status**: needs-research
**Type**: quality
**Paper**: NSNQuant (NeurIPS 2025, arXiv:2505.18231)
**What**: Normalize-Shift-Normalize aligns token distributions to standard normal, enabling a single reusable codebook across all layers without calibration. Calibration-free. Up to 3x throughput.
**How it applies**: Replace our normalize → rotate → quantize pipeline with normalize → shift → normalize → rotate → quantize. Could make codebooks even more universal (they already are, but this might help edge cases).
**Difficulty**: Low (a few days to test).

### 23. Attention-sink token protection
**Status**: needs-research
**Type**: quality
**Paper**: AnTKV (arXiv:2506.19505)
**What**: Keep first few tokens (attention sinks) and most recent tokens at full precision. Only ~1% of tokens need protection. Significant quality improvement for multi-turn conversations and system prompts.
**How it applies**: In `llama-kv-cache.cpp`, use q8_0 for KV entries at positions 0..K and positions (seq_len-M)..seq_len, turbo for the rest. K and M are small constants (e.g. 4-16).
**Difficulty**: Low-Medium (1 week). High impact for chat/instruction-following quality.

### 24. Per-head adaptive precision
**Status**: needs-research
**Type**: quality
**Papers**: KVC-Q (ScienceDirect 2026), KVTuner (ICML 2025), MixKVQ (arXiv:2512.19206)
**What**: Different attention heads have different quantization sensitivity. "Retrieval heads" (sparse, peaked attention) are sensitive; "streaming heads" (diffuse attention) are robust. Allocate turbo4 to sensitive heads, turbo3 to robust ones. Same average bit rate, better quality.
**Challenge**: Need per-head type in KV cache allocation — currently per-layer only. Significant infra change.
**Difficulty**: High (2-3 weeks). Meaningful quality gain at same compression.

### 25. Drop QJL entirely (turbo3-only approach)
**Status**: needs-research — **validated by TheTom** (QJL unnecessary, PPL matched, faster/simpler)
**Type**: simplification + speed
**Source**: TheTom's `turboquant_plus` (220 stars) + direct confirmation from Tom
**What**: turbo4 uses 3-bit codebook + 1-bit QJL signs. Tom validated that dropping QJL and giving all bits to the codebook (4-bit Lloyd-Max, no QJL) is faster and equivalent quality. Block size 32 beats 128 for FA parallelism.
**How it applies**: Would create a new type (turbo4_noqjl?) with 4-bit codebook indices, block size 32, no QJL overhead. Simpler dequant (just codebook lookup + norm multiply, no sign correction).
**Difficulty**: Medium (1-2 weeks). Could replace turbo4 if benchmarks confirm.
**Note**: Our turbo4 with norm correction currently *beats* q8_0 PPL. Need to check if dropping QJL + norm correction still beats it, or if QJL+correction is what's giving us that edge.

---

## Needs Research — Architecture / New Formats

### 26. CommVQ — RoPE-commutative codebooks
**Status**: needs-research
**Type**: architecture
**Paper**: CommVQ (ICML 2025, arXiv:2506.18879, Apple/UMass)
**What**: Codebook trained via EM to commute with RoPE: `RoPE(codebook[i]) = codebook[RoPE_perm(i)]`. Eliminates need for pre-rotate-queries entirely. 87.5% KV reduction at 2-bit.
**How it applies**: Would replace our FWHT rotation + pre-rotate-queries approach. The codebook itself handles the rotation. Eliminates the TURBO_WHT graph op and shared memory Q rotation.
**Difficulty**: Very High. Requires EM-trained per-model codebooks, new quantization pipeline, changes to codebook storage.

### 27. ~~ConvRot — group rotation instead of full-dim FWHT~~
**Status**: dropped — **failed by TheTom** (group-32 rotation: PPL 7.06 vs target 6.19)
**Paper**: ConvRot (arXiv:2512.03673, Dec 2025)
**What**: Replace full d=128 FWHT with group-of-32 Hadamard transforms. Tom tested this directly and it produces unacceptable PPL. Full d=128 rotation is necessary for proper decorrelation.

### 28. turbo2 (2-bit) and turbo5 (5-bit) variants
**Status**: needs-research
**Type**: new formats
**What**: turbo2 = ~6x compression (aggressive, for very long contexts). turbo5 = nearly lossless. RotateKV achieves <0.3 PPL degradation at 2-bit, suggesting turbo2 is viable with channel reordering (#19).
**Difficulty**: Medium (1-2 weeks per variant).

### 29. Blackwell native FP4/FP6 tensor cores
**Status**: needs-research (hardware dependent)
**Type**: speed (future)
**Paper**: NVIDIA Blackwell `tcgen05.mma` with mixed FP4/FP6/FP8 inputs
**What**: On B200/RTX5090, turbo3's 3-bit values could be zero-padded to 4-bit and use native FP4 tensor cores. Eliminates dequant bottleneck entirely for Q*K matmul.
**Difficulty**: Medium (when targeting Blackwell). Long-term path.

### 30. Dynamic quantization switching at VRAM thresholds
**Status**: needs-research
**Type**: quality + memory
**What**: Start with q8_0, auto-switch to turbo when VRAM pressure rises. LogQuant (arXiv:2503.19950) shows attention spikes follow log distribution — recent tokens need more precision, older tokens can be compressed more aggressively. PM-KVQ explores progressive bit-width lowering per block.
**Difficulty**: High (2-3 weeks).

### 31. Turbo types in speculative decoding draft model
**Status**: ready
**Type**: speed + memory
**What**: Draft models use `-ctkd`/`-ctvd` flags. turbo4 on draft KV saves VRAM, allowing larger draft caches. Drafts are tolerant of quant noise since outputs are verified.
**TODO**: Benchmark acceptance rate and overall tok/s with turbo4 draft KV.

### 32. Fused quantization in QKV projection
**Status**: needs-research
**Type**: speed (prefill)
**Paper**: TurboAttention FlashQ (Microsoft, arXiv:2412.08585)
**What**: Fuse turbo quantization into the QKV projection pass rather than as a separate SET_ROWS step. Avoids materializing full-precision KV before quantization.
**Difficulty**: High. Deep integration with ggml compute graph.

### 33. Entropy coding for stored/offloaded caches
**Status**: needs-research
**Type**: compression (storage)
**Paper**: KVTC (NVIDIA, ICLR 2026, arXiv:2511.01815)
**What**: Lloyd-Max codebook indices aren't uniformly distributed — indices near zero are more common. Arithmetic/Huffman coding could save ~0.3-0.5 bits/value, pushing turbo3 from 4.9x to ~6x compression. Apply when caching to disk/CPU for prefix sharing.
**Difficulty**: Medium (1-2 weeks). Only helps storage, not in-GPU decode.

### 34. Cross-layer codebook sharing
**Status**: needs-research
**Type**: compression
**Paper**: XQuant (arXiv:2510.11236)
**What**: Exploit redundancy across layers. If FWHT normalizes distributions well enough, a single codebook works for all layers (which we already do). But cross-layer *delta coding* — encode layer N's KV as delta from layer N-1 — could push compression further.
**Difficulty**: High. Complex dependency chain.

### 35. HCAttention — values on CPU, keys on GPU
**Status**: needs-research
**Type**: memory (extreme context)
**Paper**: HCAttention (arXiv:2507.19823)
**What**: Keep keys on GPU for scoring, offload values to CPU, fetch only selected values (top-k attention positions). Enables 4M token context on single A100.
**How it applies**: With turbo-compressed keys on GPU (tiny footprint), you could score against millions of cached tokens and only fetch the needed values from CPU. Extreme long-context scenario.
**Difficulty**: Very High. Major architectural change.

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

### Ecosystem (as of 2026-03-26)
- **TheTom/llama-cpp-turboquant** (23 stars, 8 forks) — Metal GPU, upstream for this repo
- **TheTom/turboquant_plus** (220 stars) — Python reference, dropped QJL, documents 739→2747 tok/s journey
- **tonbistudio/turboquant-pytorch** (253 stars) — Full PyTorch + Triton, validated on Qwen2.5-3B
- **Dejan.ai** — Fused Triton kernel, 1.18-1.22x speedup, pre-rotate-queries trick
- **Aaryan-Kapoor** — CPU-only TQ3_0 in llama.cpp
- **veritatisquaesitoressumus** — CPU complete in ik_llama.cpp, CUDA awaiting validation
- **mudler/LocalAI** — Experimenting, no PR yet
- **vLLM #38171** — Working PoC from vllm-omni team, joint PR in progress
- **Mainline llama.cpp** — Discussion #20969 (multi-contributor), no merged PR yet

### Key papers
- TurboQuant (Google, ICLR 2026) — the original
- PolarQuant (Google, arXiv:2502.02617) — same authors, polar coordinate decomposition
- RotateKV (IJCAI 2025) — channel reordering + FWHT
- BitDecoding (HPCA 2026) — TC-accelerated low-bit KV decode
- SageAttention 1/2/3 (ICLR/ICML/NeurIPS 2025) — INT8/INT4 attention
- KVTuner (ICML 2025) — per-layer/head sensitivity analysis
- WUSH (arXiv:2512.00956) — optimal transform theory
- NSNQuant (NeurIPS 2025) — calibration-free normalization
- CommVQ (ICML 2025) — RoPE-commutative codebooks
- KVTC (NVIDIA, ICLR 2026) — 20x compression via transform coding
- Kitty (MLSys 2026) — uniform-precision tensor decomposition
- SmoothRot (arXiv:2506.05413) — channel scaling before Hadamard
- ConvRot (arXiv:2512.03673) — group Hadamard as convolution
- TurboAttention (Microsoft) — fused quant + FlashQ + SAS softmax
- HadaCore (arXiv:2412.08832) — TC-accelerated FWHT, 1.1-3.5x speedup
- AnTKV (arXiv:2506.19505) — attention-sink protection
- "More Keys Less Values" (arXiv:2502.15075) — asymmetric K/V theory

---

## Dropped

### Group-32 rotation (ConvRot)
**Reason**: TheTom tested directly. PPL 7.06 vs target 6.19. Full d=128 FWHT rotation is necessary for proper decorrelation. Smaller group sizes lose too much.

### Custom GGML_OP_TURBO_WHT as speed optimization
**Reason**: TheTom found it's a red herring — same speed as dense matmul. The graph-level op works functionally but doesn't help performance. We keep it for correctness (Q pre-rotation, V un-rotation) but shouldn't expect speed gains from it.

### Gemini's RoPE/WHT commutativity theory
**Reason**: TheTom investigated, wasn't the actual root cause of quality issues. The real constraint is simpler: WHT must be applied after RoPE, which our implementation does correctly.
