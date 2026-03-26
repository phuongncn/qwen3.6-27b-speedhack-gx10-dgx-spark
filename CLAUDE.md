IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Project conventions

- **Save all benchmark results** to `benchmark-results.md` as they come in. Benchmarks take a long time to run and will be lost during context compaction. Always append new results to this file.
- **Track experiments** in `experiments.md` — update status as experiments are tested.
- The benchmark server is `ssh root@dorei` (RTX 3090 24GB). Build at `/root/llama-cuda-turbo/build/bin/`. Model at `/root/Qwen3.5-27B-heretic.Q6_K.gguf`.
- Use `scp` to deploy code changes from local to server, then rebuild on server.
- Server cmake flags: `cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86`
- When running llama-perplexity, grep for `"Final estimate"` to get PPL values.
- When running llama-bench, grep for `"tg64"` to get decode speed values.
- Run benchmarks sequentially on the GPU (not in parallel) to avoid contention.

## Experiment workflow

- **Create a branch** for each experiment: `git checkout -b experiment/<name>`
- **Commit working state** on the experiment branch when results are valid
- **Always verify PPL** against a fresh build of committed HEAD before concluding an experiment helps or hurts
- **Merge into main** (`feature/turboquant-kv-cache`) only when results are confirmed better
- **Never disable working features** on main without first verifying the change on an experiment branch
