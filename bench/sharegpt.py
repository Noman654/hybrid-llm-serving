"""sharegpt.py OUT.json (needs SHAREGPT=path/to/ShareGPT_V3_unfiltered_cleaned_split.json): ShareGPT serving mix (real text). First human turn, chat template, output length = reference reply
length (tokens, clipped 32..1024), fixed via min_tokens+ignore_eos so every config does identical work.
Env: B, MNS, CGMAX, SPEC (MTP draft tokens), GDN_REPLAY, GMU. Reports total time and spec-decode acceptance."""
import os, sys, glob, json, time, random
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kernels", "replay"))
import replay_patch  # GDN_REPLAY=1 enables the replay kernel; default off
from vllm import LLM, SamplingParams, TokensPrompt
M = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")  # HF id or local path
B = int(os.environ.get("B", "1024"))
KW = dict(max_model_len=4096, max_num_seqs=int(os.environ.get("MNS", "256")), enable_prefix_caching=False, mamba_ssm_cache_dtype="bfloat16",
          gpu_memory_utilization=float(os.environ.get("GMU", "0.88")), generation_config="vllm", language_model_only=True, disable_log_stats=False)
if os.environ.get("CGMAX"): KW["compilation_config"] = {"max_cudagraph_capture_size": int(os.environ["CGMAX"])}
if os.environ.get("SPEC"): KW["speculative_config"] = {"method": "mtp", "num_speculative_tokens": int(os.environ["SPEC"])}
llm = LLM(M, **KW)
tok = llm.get_tokenizer()
d = json.load(open(os.environ["SHAREGPT"])); rng = random.Random(0); rng.shuffle(d)
P, SP = [], []
for x in d:
    c = x.get("conversations", [])
    if len(c) < 2 or c[0]["from"] != "human" or c[1]["from"] != "gpt": continue
    ids = tok.apply_chat_template([{"role": "user", "content": c[0]["value"]}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"): ids = ids["input_ids"]
    if not (16 <= len(ids) <= 2048): continue
    ol = min(1024, max(32, len(tok(c[1]["value"]).input_ids)))
    P.append(TokensPrompt(prompt_token_ids=list(ids))); SP.append(SamplingParams(max_tokens=ol, min_tokens=ol, ignore_eos=True, temperature=0))
    if len(P) == B: break
llm.generate(P[:8], SamplingParams(max_tokens=16), use_tqdm=False)
def spec():
    m = {x.name: x for x in llm.get_metrics()}
    g = lambda n: getattr(m.get(n), "value", 0) if n in m else 0
    return g("vllm:spec_decode_num_drafts"), g("vllm:spec_decode_num_draft_tokens"), g("vllm:spec_decode_num_accepted_tokens")
R = {"prompt_tok": sum(len(p["prompt_token_ids"]) for p in P), "out_tok": sum(s.max_tokens for s in SP)}
for k in range(2):
    s0 = spec(); t0 = time.time(); llm.generate(P, SP, use_tqdm=False); dt = time.time() - t0; s1 = spec()
    nd, ndt, na = (b - a for a, b in zip(s0, s1))
    R[str(k)] = {"total_s": dt, "accept_len": (1 + na / nd) if nd else None, "accept_rate": (na / ndt) if ndt else None}
    print("RESULT", k, R[str(k)], flush=True)
json.dump(R, open(sys.argv[1], "w"))
