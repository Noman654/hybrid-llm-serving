"""gdn_prefill_accuracy.py: accuracy + speed of chunked GDN prefill variants vs an fp64 token-by-token reference.

Shapes match one Qwen3.8-27B GDN layer: 16 key heads, 48 value heads, K = V = 128, bf16 inputs.
Variants:
  fla       vLLM's Triton/FLA chunk_gated_delta_rule (triangular inverse stored in bf16)   <- ROCm/AMD path
  fla_ai32  same, but the triangular inverse kept in fp32 and applied in fp32 (wy_fast32.py)
  fi        FlashInfer chunk_gated_delta_rule (vLLM's default on H100)
Reports max/mean abs error of outputs and final state vs fp64, and median kernel time.
Env: T (tokens, default 4096), NSEQ (sequences, default 4), SEED.
"""
import os, sys, math, statistics
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm.third_party.flash_linear_attention.ops.chunk as CH
from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
import wy_fast32

dev = "cuda"; torch.manual_seed(int(os.environ.get("SEED", "0")))
T, NSEQ = int(os.environ.get("T", "4096")), int(os.environ.get("NSEQ", "4"))
Hk, Hv, K, V = 16, 48, 128, 128
N = T * NSEQ
cu = torch.arange(0, N + 1, T, device=dev, dtype=torch.int32)
q = torch.randn(1, N, Hk, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, N, Hk, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, N, Hv, V, device=dev, dtype=torch.bfloat16)
# realistic gates: log-decay in (-1, 0), beta in (0, 1)
g = (-torch.rand(1, N, Hv, device=dev) * 0.9 - 0.01).to(torch.float32)
beta = torch.sigmoid(torch.randn(1, N, Hv, device=dev)).to(torch.bfloat16)
h0 = torch.zeros(NSEQ, Hv, K, V, device=dev, dtype=torch.float32)
scale = K ** -0.5
qn, kn = l2norm_fwd(q), l2norm_fwd(k)


def reference():
    """fp64 recurrence, one token at a time. State S[h] is K x V."""
    rep = Hv // Hk
    Q = qn[0].double().repeat_interleave(rep, 1); Kt = kn[0].double().repeat_interleave(rep, 1)
    Vv, G, Bt = v[0].double(), g[0].double(), beta[0].double()
    O = torch.empty(N, Hv, V, device=dev, dtype=torch.float64); S_fin = []
    for s in range(NSEQ):
        S = torch.zeros(Hv, K, V, device=dev, dtype=torch.float64)
        for t in range(s * T, (s + 1) * T):
            S = S * torch.exp(G[t])[:, None, None]
            kv = torch.einsum("hk,hkv->hv", Kt[t], S)
            S = S + Bt[t][:, None, None] * Kt[t][:, :, None] * (Vv[t] - kv)[:, None, :]
            O[t] = torch.einsum("hk,hkv->hv", Q[t], S) * scale
        S_fin.append(S)
    return O, torch.stack(S_fin)


def run_fla(ai32=False):
    orig = (CH.solve_tril, CH.recompute_w_u_fwd)
    if ai32:
        CH.solve_tril = lambda **kw: orig[0](**{**kw, "output_dtype": torch.float32})
        CH.recompute_w_u_fwd = wy_fast32.recompute_w_u_fwd
    try:
        o, st = chunk_gated_delta_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=h0.clone(),
                                       output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    finally:
        CH.solve_tril, CH.recompute_w_u_fwd = orig
    return o[0], st


def run_fi():
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi
    o, st = fi(q=qn[0].contiguous(), k=kn[0].contiguous(), v=v[0].contiguous(), g=torch.exp(g[0]).contiguous(),
               beta=beta[0].float().contiguous(), scale=scale, initial_state=h0.clone(), output_final_state=True,
               cu_seqlens=cu.to(torch.int64), backend="flashinfer")
    return o, st


def timeit(f, n=20):
    for _ in range(3): f()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return statistics.median(ts)


O_ref, S_ref = reference()
print(f"T={T} x {NSEQ} seqs, Hk={Hk} Hv={Hv} K=V={K}; |O_ref| mean {O_ref.abs().mean():.3e}, |S_ref| mean {S_ref.abs().mean():.3e}")
for name, f in [("fla", lambda: run_fla(False)), ("fla_ai32", lambda: run_fla(True)), ("fi", run_fi)]:
    try:
        o, st = f()
    except Exception as e:
        print(f"{name:9s} FAILED: {e!r}"[:300]); continue
    eo = (o.double() - O_ref).abs()
    st = st.double()
    es = (st - S_ref).abs()
    es_t = (st.transpose(-1, -2) - S_ref).abs()
    if es_t.max() < es.max(): es = es_t  # state layout may be [V, K]
    ms = timeit(f)
    print(f"{name:9s} out max {eo.max():.3e} mean {eo.mean():.3e} | state max {es.max():.3e} mean {es.mean():.3e} | {ms:.3f} ms")
