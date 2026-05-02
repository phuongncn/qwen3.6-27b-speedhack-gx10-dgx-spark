# Qwen3.6-27B × DFlash — 4× Faster on NVIDIA DGX Spark (GB10)

**7–11 tok/s → 38-40 tok/s coding. 4× speedup. Same hardware. Same model.**

<p align="center">
  <img src="buunslamma.png" alt="buun llama" width="200"/>
</p>

## The Story

It started with a single number: **7 tok/s**.

Running Qwen3.6 27B on the ASUS GX10 (NVIDIA DGX Spark / GB10 Grace-Blackwell) with stock llama.cpp. Tried everything — TurboQuant KV cache, CUDA graph optimizations, Flash Attention, tweaking batch sizes, thread pinning, every environment variable in the book. Best I got was ~10 tok/s on a good day. For a machine with 128GB unified memory and a Blackwell GPU, that felt wrong.

Before that, I actually found **vLLM + MTP** first. It also hit around 30 tok/s — competitive numbers. But vLLM takes minutes to load the model, and the speed isn't stable — it surges and dips unpredictably. I've always loved llama.cpp for loading models in seconds and just getting to work. That fast startup matters when you're iterating, testing, rebooting, switching models. vLLM felt like waiting for a server to boot; llama.cpp felt like launching an app.

Then I stumbled across **DFlash** — block-diffusion speculative decoding. A tiny 1.7GB draft model that generates *multiple tokens per forward pass*, verified in bulk by the target model. The spiritbuun fork had already done the heavy lifting: GPU cross-ring, tape replay, fused Gated Delta Net kernels.

But the out-of-the-box numbers weren't great on long context. Acceptance rate would collapse to ~12% beyond 3K tokens. Draft tokens were being generated but mostly rejected — wasted compute.

**Days of testing.** Hundreds of requests. Reading speculative decoding logs cycle by cycle. Then the breakthrough: `p_min` — a confidence threshold that stops drafting the moment the draft model loses confidence. Combined with adaptive draft length that caps `n_max` at long context.

The result: **acceptance rate jumped from 39% to 67%**. Same hardware. Same models. Just smarter drafting.

## Benchmarks

All tests on GB10, identical prompts, temperature=0.0 (greedy). Stock uses q8_0 KV cache, DFlash uses turbo4.

### Qwen3.6-27B (Dense) — DFlash vs Stock

| Scenario | Stock (no DFlash) | DFlash (optimized) | Speedup |
|----------|:-----------------:|:------------------:|:-------:|
| HTML/JS coding (400 tok) | 7-11 tok/s | **38-40 tok/s** | **~4×** |
| Python coding (400-500 tok) | 7-11 tok/s | **24-25 tok/s** | **~2.5×** |
| Short chat (150 tok) | 7-11 tok/s | **23-25 tok/s** | **~2.5×** |
| Medium context (300 tok) | 7-11 tok/s | **20-22 tok/s** | **~2×** |
| Sustained 2048 tok | 7-11 tok/s | **27-29 tok/s** | **~2.5×** |

Stock llama.cpp is a consistent 7-11 tok/s regardless of scenario. DFlash adds 2-4× speedup depending on content type and context length.

### Target Model Quantization Comparison (all with DFlash + Q8_0 draft)

| Scenario | Q4_K_M (16GB) | Q8_0 (27GB) | BF16 (51GB) |
|----------|:-------------:|:-----------:|:-----------:|
| HTML/JS coding (400 tok) | **38-40 tok/s** | 21-23 tok/s | 15-17 tok/s |
| Short chat (150 tok) | **23-25 tok/s** | 20-22 tok/s | 14-16 tok/s |
| Medium context (300 tok) | **20-22 tok/s** | 13-15 tok/s | 8-10 tok/s |
| Sustained 2048 tok | **27-29 tok/s** | 23-25 tok/s | 13-15 tok/s |
| Accept rate | 52–61% | 52–71% | 59% |

**Q4_K_M is the clear winner** — 18-29% faster than Q8_0, ~2× faster than BF16. GB10's 273 GB/s unified memory bandwidth is the bottleneck: larger models spend more time reading weights per token. Q4_K_M quality is near-lossless for all practical use.

### What Affects Speed

- **Content type** — HTML/JS/CSS drafts faster than Python (38-40 vs 23-25 tok/s). Boilerplate patterns (tags, brackets, repeated structures) are easier for the draft model to predict.
- **Prompt length** — longer context slightly reduces speed but p_min keeps acceptance rate stable.
- **Temperature** — negligible impact (temp=0.0 vs 0.7: within 1-2 tok/s).
- **Model size** — every 10GB of extra model weights costs ~5-7 tok/s on GB10.

**TL;DR:** Q4_K_M target + Q8_0 draft + `--spec-draft-n-max 14` = best combination. 38-40 tok/s HTML/JS, 23-25 tok/s chat.

## What Makes This Fast

- **DFlash (Block-Diffusion Speculative Decoding)** — draft model generates multiple tokens per forward pass, not one-at-a-time like traditional speculative decoding
- **`p_min` confidence threshold** — aborts drafting on low-confidence tokens, doubling effective acceptance rate
- **Adaptive draft length** — automatically reduces draft tokens at long context where acceptance drops
- **GPU cross-ring + tape replay** — spiritbuun's custom CUDA extensions for low-latency draft verification
- **TurboQuant KV cache (turbo4)** — 3.8× KV cache compression, near-lossless
- **Fused Gated Delta Net** — optimized kernel for Qwen3.6's recurrent state (~75% of layers)
- **Runtime ubatch switching** — large ubatch for prefill, small ubatch for decode

## Hardware

Tested on **NVIDIA DGX Spark (ASUS GX10)**:
- Grace-Blackwell GB10 superchip
- 128GB unified memory (LPDDR5X, 273 GB/s)
- Blackwell GPU: SM 12.1, 99KB shared memory per block
- Ubuntu 24.04, CUDA 13.0

## Quick Start

### 1. Clone & Build

```bash
git clone https://github.com/phuongncn/qwen3.6-27b-speedhack-gx10-dgx-spark.git
cd qwen3.6-27b-speedhack-gx10-dgx-spark
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_RPC=OFF
cmake --build . -j20 --config Release
```

### 2. Download Models

**Target model** (Q4_K_M, 16GB):
```
https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
→ Qwen3.6-27B-Q4_K_M.gguf
```

**Draft model** (Q8_0, 1.7GB):
```
https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF
→ Qwen3.6-27B-DFlash-Q8_0.gguf
```

### 3. Launch Server

```bash
./build/bin/llama-server \
  -m Qwen3.6-27B-Q4_K_M.gguf \
  -md Qwen3.6-27B-DFlash-Q8_0.gguf \
  --spec-type dflash \
  --spec-dflash-default \
  --spec-draft-p-min 0.3 \
  --spec-draft-n-max 14 \
  --spec-draft-n-min 0 \
  --draft-max 14 \
  -ngl 99 -ngld 99 \
  -c 256000 -cd 256 \
  -b 2048 -ub 1024 \
  -ctk turbo4 -ctv turbo4 \
  -fa 1 \
  --host 0.0.0.0 --port 8080 \
  --jinja \
  --reasoning off \
  --cache-ram 0
```

### 4. Use

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Explain quantum computing in simple terms"}], "max_tokens":200}'
```

OpenAI-compatible API — works with any client (Open WebUI, Continue.dev, RooCode, etc.)

## Key Flags Explained

| Flag | Value | What it does |
|------|-------|---------------|
| `--spec-dflash-default` | — | Enables DFlash optimized defaults |
| `--spec-draft-p-min` | 0.3 | Stop drafting when confidence < 0.3 (doubles accept rate) |
| `--spec-draft-n-max` | 14 | Max draft tokens, auto-reduces at long context |
| `--spec-draft-n-min` | 0 | Minimum draft tokens (0 = fully adaptive) |
| `--draft-max` | 14 | Max tokens per draft forward pass |
| `-ctk` / `-ctv` | turbo4 | TurboQuant scalar KV cache (3.8× compression) |
| `-cd` | 256 | Draft model context (small = fast, drafter only sees recent tokens) |

## Bonus: Qwen3.6-35B-A3B (MoE) — 92-130 tok/s coding

The 35B-A3B is a hybrid SSM+Attention model. A one-line fix to the SSM conv state rebuild after draft rejection unlocks clean output at 2× stock speed.

| Scenario | Stock (no DFlash) | DFlash (optimized) | Speedup |
|----------|:-----------------:|:------------------:|:-------:|
| HTML/JS coding (short, ~600 tok) | 60-66 tok/s | **92-101 tok/s** | **~1.5×** |
| HTML/JS coding (sustained, ~2000 tok) | 60-66 tok/s | **85-92 tok/s** | **~1.4×** |
| Short chat (~100 tok) | 60-66 tok/s | **40-50 tok/s** | ~0.7× |

**Target model:** [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)

**Draft model:** No pre-built GGUF — convert from [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash) safetensors:

```bash
# 1. Download the z-lab draft model (safetensors + config)
huggingface-cli download z-lab/Qwen3.6-35B-A3B-DFlash \
  --local-dir /path/to/Qwen3.6-35B-A3B-DFlash

# 2. Copy tokenizer files from the original Qwen repo (z-lab doesn't ship them)
huggingface-cli download Qwen/Qwen3.6-35B-A3B \
  tokenizer.json tokenizer_config.json \
  --local-dir /path/to/Qwen3.6-35B-A3B-DFlash

# 3. Install dependencies and convert to BF16 GGUF
pip install gguf sentencepiece
python3 convert_hf_to_gguf.py \
  /path/to/Qwen3.6-35B-A3B-DFlash \
  --outtype bf16 \
  --outfile Qwen3.6-35B-A3B-DFlash-BF16.gguf
```

## Credits

This project stands on the shoulders of:

- **[spiritbuun](https://github.com/spiritbuun/buun-llama.cpp)** — creator of the DFlash llama.cpp fork, GPU cross-ring, tape replay, and the Qwen3.6 DFlash draft model
- **[ggml.ai](https://github.com/ggerganov/llama.cpp)** — the llama.cpp ecosystem that makes all of this possible
- **[z-lab](https://huggingface.co/z-lab)** — Qwen3.6-27B-DFlash safetensors model
- **[unsloth](https://huggingface.co/unsloth)** — Qwen3.6-27B GGUF quantizations
- **[NVIDIA](https://www.nvidia.com)** — GB10 Grace-Blackwell hardware platform
- **[FlashQLA](https://github.com/flashinfer-ai/FlashQLA)** — explored but incompatible with GB10 (99KB shared memory limit vs 192KB required)

## License

MIT — see [LICENSE](LICENSE). Built on llama.cpp which is also MIT-licensed.
