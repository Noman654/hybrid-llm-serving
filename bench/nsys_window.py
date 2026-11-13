"""nsys_window.py: serving mix (1024 reqs, 256 slots, PL 512, varied output) with a 4 s profiler window in steady state.
Run under: nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -o out python nsys_window.py
then: nsys export --type sqlite out.nsys-rep && python gaps.py out.sqlite"""
import os, sys, time, threading, glob, random
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "replay"))
import replay_patch  # GDN_REPLAY=1 enables the replay kernel; default off
import torch
from vllm import LLM, SamplingParams, TokensPrompt
M = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")  # HF id or local path
KW = dict(max_model_len=4096, max_num_seqs=256, enable_prefix_caching=False, mamba_ssm_cache_dtype="bfloat16",
          gpu_memory_utilization=0.88, generation_config="vllm", language_model_only=True)
if os.environ.get("CGMAX"): KW["compilation_config"] = {"max_cudagraph_capture_size": int(os.environ["CGMAX"])}
if replay_patch.ON and getattr(replay_patch, "NEEDS_RSSM", True): KW["use_replayssm"] = True
llm = LLM(M, **KW)
rng = random.Random(0); OL = 1024
P = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000) for _ in range(512)]) for _ in range(1024)]
_r = random.Random(1); sp = [SamplingParams(max_tokens=(o := _r.randint(OL // 8, OL)), min_tokens=o, ignore_eos=True, temperature=0) for _ in range(1024)]
llm.generate(P[:8], SamplingParams(max_tokens=16, ignore_eos=True), use_tqdm=False)
t = threading.Thread(target=lambda: llm.generate(P, sp, use_tqdm=False)); t.start()
time.sleep(float(os.environ.get("DELAY", "60"))); torch.cuda.profiler.start(); time.sleep(4); torch.cuda.profiler.stop(); t.join(); print("NSYS_GEN_DONE", flush=True)
