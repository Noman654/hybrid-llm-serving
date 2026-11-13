"""ladder.py OUT.json: one serving benchmark, one GPU. B prompts of PL random tokens through MNS slots, exactly OL output tokens
each (OLVAR=1: per-request output length uniform in [OL/8, OL], seeded, so every config does identical work). Reports total
time per run (2 runs). Knobs: PC (prefix caching 0/1), SSMDT (GDN state dtype: auto=fp32 | bfloat16), CGMAX (CUDA-graph
capture limit), GDN_REPLAY (replay kernel 0/1), MNS, GMU, STEPLOG (record tokens per engine step). See run_ladder.sh."""
import os, sys, glob, json, time, random
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = os.environ.get("MP", "0"); os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "replay"))
import replay_patch  # GDN_REPLAY=1 enables the replay kernel; default off
from vllm import LLM, SamplingParams, TokensPrompt
B, PL, OL = int(os.environ.get("B", "256")), int(os.environ.get("PL", "1000")), int(os.environ.get("OL", "512"))
M = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")  # HF id or local path
KW = dict(max_model_len=4096, max_num_seqs=int(os.environ.get("MNS", max(B, 8))), enable_prefix_caching=os.environ.get("PC","0")=="1", mamba_ssm_cache_dtype=os.environ.get("SSMDT","bfloat16"),
          gpu_memory_utilization=float(os.environ.get("GMU", "0.88")), generation_config="vllm", language_model_only=True)
if replay_patch.ON and getattr(replay_patch, "NEEDS_RSSM", True): KW["use_replayssm"] = True
if os.environ.get("CGMAX"): KW["compilation_config"] = {"max_cudagraph_capture_size": int(os.environ["CGMAX"])}
if os.environ.get("CG64"):  # mimic vLLM PR #64825: default sizes up to 512, then every 64 tokens up to CG64
    _sz = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16)) + list(range(576, int(os.environ["CG64"]) + 1, 64))
    KW["compilation_config"] = {"cudagraph_capture_sizes": _sz}
if os.environ.get("MNBT"): KW["max_num_batched_tokens"] = int(os.environ["MNBT"])
STEPS = []
if os.environ.get("STEPLOG"):
    from vllm.v1.worker import gpu_model_runner as _G
    _oe = _G.GPUModelRunner.execute_model
    def _ex(self, so, *a, **k):
        r = _oe(self, so, *a, **k)
        STEPS.append((time.perf_counter(), so.total_num_scheduled_tokens, len(so.num_scheduled_tokens)))
        return r
    _G.GPUModelRunner.execute_model = _ex
llm = LLM(M, **KW)
rng = random.Random(0)
P = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000) for _ in range(PL)]) for _ in range(B)]
gen = SamplingParams(max_tokens=OL, min_tokens=OL, ignore_eos=True, temperature=0)
if os.environ.get("OLVAR"):  # serving mix: per-request output length uniform in [OL/8, OL], seeded -> identical work in every run
    _r = random.Random(1); gen = [SamplingParams(max_tokens=(o := _r.randint(OL // 8, OL)), min_tokens=o, ignore_eos=True, temperature=0) for _ in range(B)]
one = SamplingParams(max_tokens=1, temperature=0)
llm.generate(P[:8], gen[:8] if isinstance(gen, list) else gen, use_tqdm=False)
res = {}
for rep in range(2):
    t0 = time.time(); llm.generate(P, one, use_tqdm=False); tp = time.time() - t0
    t0 = time.time(); o = llm.generate(P, gen, use_tqdm=False); tg = time.time() - t0
    n = sum(len(x.outputs[0].token_ids) for x in o)
    res[rep] = dict(prefill_s=tp, total_s=tg, tok=n, decode_ms_per_step=(tg - tp) / (OL - 1) * 1e3)
print(f"RESULT replay={replay_patch.ON} B={B} PL={PL} OL={OL} " + " | ".join(f"total {r['total_s']:.2f}s prefill {r['prefill_s']:.2f}s decode {r['decode_ms_per_step']:.2f} ms/step" for r in res.values()), flush=True)
res["steps"] = STEPS
json.dump(res, open(sys.argv[1], "w"))
