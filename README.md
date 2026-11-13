# hybrid-llm-serving

Code and write-up for **[Where Did the Time Go? Serving a Hybrid LLM on H100](https://noman654.github.io/hybrid-llm-serving/)**: a case study of serving Qwen3.8-27B-FP8 (48 Gated DeltaNet layers + 16 attention layers) on vLLM 0.31.

| Result (one H100 unless noted) | |
|---|---|
| Stock vLLM → bf16 GDN state → prefix caching off → CUDA-graph limit 2,048 → replay kernel | 210.1 s → 122.5 s (**1.72×**) on a serving mix without shared prefixes |
| CUDA-graph limit 2,048 alone, 8×H100, already-tuned config, ShareGPT | **+10.4%** output tok/s |
| Fused GDN conv1d + prep kernel | −30% kernel time, +0.8% prefill, **bit-exact** vs stock |

The post explains why each change helps, what didn't work, and how the numbers were checked.

## Layout

```
kernels/
  fconv.py              fused causal conv1d + l2norm + q/k/v split for GDN prefill (bit-exact vs vLLM 0.31)
  test_fconv.py         bit-exactness + timing vs stock, both conv-state layouts, mixed decode/prefill batches
  replay/               replay (deferred state write) GDN decode kernel + runtime patch for vLLM 0.31
bench/
  ladder.py, run_ladder.sh   the Section 6 ladder (one serving benchmark, one change at a time)
  gsm8k.py, mcnemar.py       full GSM8K (1,319 questions) + per-question McNemar test vs a baseline run
  sharegpt.py                ShareGPT serving mix (real text, fixed output lengths)
  prefill.py                 prefill throughput + greedy-output fingerprint, with/without the fused kernel (FCONV=1)
  nsys_window.py, gaps.py    4 s nsys window in steady-state serving -> GPU busy/idle and gap histogram
  kernel_breakdown.py        kernel-time shares from an nsys sqlite export
  gdn_prefill_accuracy.py    chunked GDN prefill (FLA / FlashInfer) vs an fp64 token-by-token reference
serve/serve.sh          the 8xH100 serving config from the post's appendix
figures/gen_figs.py     regenerates the post's SVG figures
docs/                   the post (GitHub Pages)
```

## Requirements

- vLLM **0.31.0** (the kernels patch vLLM internals at runtime and are version-specific), PyTorch with CUDA, an H100 (80 GB).
- `MODEL` defaults to `Qwen/Qwen3.8-27B-FP8`; set it to a local path if you have the weights.
- `sharegpt.py` needs `SHAREGPT=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json`.

## Quick start

```bash
# bit-exactness of the fused kernel (needs a GPU, no model weights)
python kernels/test_fconv.py

# the ladder on 5 GPUs (or GPUS="0 0 0 0 0" to run the steps one by one on one GPU)
bash bench/run_ladder.sh

# GSM8K baseline vs replay kernel, then compare per question
N=1319 MAXTOK=1024 CGMAX=2048 python bench/gsm8k.py base.json
N=1319 MAXTOK=1024 CGMAX=2048 GDN_REPLAY=1 python bench/gsm8k.py replay.json
python bench/mcnemar.py base.json replay.json
```

The single-GPU scripts run vLLM in-process with the V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`) so the runtime patches apply.

## Notes

- The replay kernel is a re-implementation of the idea in [ReplaySSM](https://tridao.me/blog/2026/replayssm/) and [DeltaLog](https://arxiv.org/pdf/2608.15533) for vLLM's GDN decode. It is research code: it needs the V1 model runner and did not beat V2 + graph limit at node scale.
- `kernels/fconv.py` and `bench/wy_fast32.py` adapt code from vLLM and flash-linear-attention; their original copyright notices are kept in the files.

## License

Apache-2.0 (see `LICENSE`).
