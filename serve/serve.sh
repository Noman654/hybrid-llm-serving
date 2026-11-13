#!/bin/bash
# 8xH100 serving config from the post's appendix: one model copy per GPU (data parallel), bf16 GDN state,
# prefix caching off (drop that flag if your traffic shares prefixes), FP8 KV cache, CUDA-graph limit 2048.
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-2400}   # graph capture takes ~7 min on 8 GPUs
exec vllm serve ${MODEL:-Qwen/Qwen3.8-27B-FP8} --port ${PORT:-8000} \
  --language-model-only --max-model-len 8192 \
  --data-parallel-size ${DP:-8} --api-server-count ${DP:-8} \
  --mamba-ssm-cache-dtype bfloat16 --no-enable-prefix-caching \
  --kv-cache-dtype fp8 --async-scheduling \
  --gpu-memory-utilization ${GMU:-0.95} --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --generation-config vllm --attention-backend FLASH_ATTN \
  --compilation-config '{"max_cudagraph_capture_size": 2048}' "$@"
