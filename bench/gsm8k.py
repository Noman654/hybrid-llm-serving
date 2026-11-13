"""gsm8k.py OUT.json: Qwen3.8-27B-FP8, V1 runner, in-process. GDN_REPLAY=0/1 selects baseline / replay kernel.
N GSM8K 5-shot prompts, greedy, MAXTOK tokens, min_tokens=1, logprobs=1 -> tokens + logprobs + accuracy.
Env: CACHE (1 = prefix caching, align mode), N, MAXTOK, MNS (max_num_seqs), EAGER."""
import os, sys, re, glob, json, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "replay"))
import replay_patch  # GDN_REPLAY=1 enables the replay kernel; default off
from vllm import LLM, SamplingParams
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; N = int(os.environ.get("N", "256")); NT = int(os.environ.get("MAXTOK", "320"))
M = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")  # HF id or local path
KW = dict(max_model_len=8192, max_num_seqs=int(os.environ.get("MNS", "256")), enable_prefix_caching=os.environ.get("CACHE", "0") == "1",
          mamba_ssm_cache_dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("GMU", "0.85")), generation_config="vllm", language_model_only=True,
          enforce_eager=os.environ.get("EAGER", "0") == "1")
if replay_patch.ON and getattr(replay_patch, "NEEDS_RSSM", True):
    KW["use_replayssm"] = True
if os.environ.get("FORCE_RSSM") == "1":  # stock kernels, but vLLM's replayssm bookkeeping on
    from vllm.config.model import ModelConfig
    ModelConfig.supports_replayssm = property(lambda self: True)
    KW["use_replayssm"] = True
if os.environ.get("ASYNC") == "0":
    KW["async_scheduling"] = False
if os.environ.get("GDNPB"): KW["gdn_prefill_backend"] = os.environ["GDNPB"]
if os.environ.get("KVDT"): KW["kv_cache_dtype"] = os.environ["KVDT"]
if os.environ.get("SPEC"): KW["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(os.environ["SPEC"])}
if os.environ.get("CGMAX"): KW["compilation_config"] = {"max_cudagraph_capture_size": int(os.environ["CGMAX"])}
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
te = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
tr = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset"))
shots = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(tr.question[:5], tr.answer[:5]))
P = [shots + f"Question: {q}\nAnswer:" for q in te.question[:N]]
gold = [a.split("####")[-1].strip().replace(",", "") for a in te.answer[:N]]
sp = SamplingParams(max_tokens=NT, temperature=0, logprobs=1, min_tokens=1, stop=["\n\nQuestion", "Question:"])
llm.generate(P[:4], sp, use_tqdm=False)  # warmup
t0 = time.time(); outs = llm.generate(P, sp, use_tqdm=False); dt = time.time() - t0
def ans(t):
    m = re.findall(r"####\s*(-?[\d,\.]+)", t) or re.findall(r"-?\d[\d,]*\.?\d*", t)
    return m[-1].replace(",", "").rstrip(".") if m else ""
res = [dict(toks=list(o.outputs[0].token_ids), lp=[float(d[t].logprob) for d, t in zip(o.outputs[0].logprobs, o.outputs[0].token_ids)],
            text=o.outputs[0].text) for o in outs]
acc = sum(ans(r["text"]) == g for r, g in zip(res, gold)) / N
ntok = sum(len(r["toks"]) for r in res)
json.dump(dict(res=res, acc=acc, secs=dt, ntok=ntok, stats=replay_patch.STATS, steps=STEPS, miss=getattr(replay_patch, "_IB", {}).get("miss_log")), open(OUT, "w"))
print(f"RESULT {OUT} replay={replay_patch.ON} cache={KW['enable_prefix_caching']} acc={acc:.4f} gen_tok={ntok} {dt:.1f}s {ntok/dt:.0f} tok/s stats={replay_patch.STATS}", flush=True)
