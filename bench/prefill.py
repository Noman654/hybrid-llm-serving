"""prefill.py: prefill throughput (N x PL, max_tokens=1) + greedy-output fingerprint, with/without fconv (env FCONV=1).
Writes token ids of a 64-prompt greedy generation (mixed lengths, 64 new tokens, batch -> mixed prefill/decode steps) to OUT."""
import os, sys, glob, random, json, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels"))
import torch
from vllm import LLM, SamplingParams, TokensPrompt
if os.environ.get("FCONV") == "1":
    import fconv; fconv.install()
M = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")  # HF id or local path
PL, N, REPS = int(os.environ.get("PL", "4096")), int(os.environ.get("N", "256")), int(os.environ.get("REPS", "3"))
llm = LLM(M, max_model_len=PL + 128, max_num_seqs=256, enable_prefix_caching=False, mamba_ssm_cache_dtype="bfloat16",
          gpu_memory_utilization=0.85, generation_config="vllm", language_model_only=True)
rng = random.Random(0)
P = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000) for _ in range(PL)]) for _ in range(N)]
llm.generate(P[:16], SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)
res = []
for r in range(REPS):
    torch.cuda.synchronize(); t0 = time.time()
    llm.generate(P, SamplingParams(max_tokens=1, ignore_eos=True), use_tqdm=False)
    torch.cuda.synchronize(); res.append(N * PL / (time.time() - t0))
# fingerprint: varied prompt lengths so steps mix prefill chunks and decodes
rng = random.Random(1)
P2 = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 100000) for _ in range(rng.randrange(20, 3000))]) for _ in range(64)]
outs = llm.generate(P2, SamplingParams(max_tokens=64, temperature=0, ignore_eos=True), use_tqdm=False)
ids = [list(o.outputs[0].token_ids) for o in outs]
json.dump(ids, open(os.environ.get("OUT", "prefill_ids.json"), "w"))
st = fconv.STATS if os.environ.get("FCONV") == "1" else {}
print(f"AB FCONV={os.environ.get('FCONV','0')} tok/s " + " ".join(f"{x:.0f}" for x in res) + f" stats={st}", flush=True)
