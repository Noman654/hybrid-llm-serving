"""test_fconv.py: bit-exactness + timing of fused conv+prep (fconv.py) vs stock causal_conv1d_fn + fused_post_conv_prep.
Qwen3.8-27B GDN shapes: dim 10240 = q 2048 + k 2048 + v 6144, width 4, 48 v-heads.
Cases: pure prefill (no initial state), mixed (64 decodes + chunked-prefill continuations + seqs shorter than state_len),
both conv-state layouts (dim-first contiguous and token-first transposed view)."""
import sys, os, statistics, types
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
import fconv

dev = "cuda"; torch.manual_seed(0)
H, KD, HV, VD, W = 16, 128, 48, 128, 4
dim = 2 * H * KD + HV * VD
NCACHE = 600
w = (torch.randn(dim, W, device=dev) * 0.5).to(torch.bfloat16)
bias = None
A_log = torch.randn(HV, device=dev); dt_bias = torch.randn(HV, device=dev)


def make_case(lens, init, dim_first):
    L = sum(lens)
    xr = torch.randn(L, dim + 7, device=dev, dtype=torch.bfloat16)[:, :dim]  # row stride != dim, like a split view
    x = xr.transpose(0, 1)  # (dim, L) channel-last
    if dim_first:
        cs = torch.randn(NCACHE, dim, W - 1, device=dev, dtype=torch.bfloat16)
    else:
        cs = torch.randn(NCACHE, W - 1, dim, device=dev, dtype=torch.bfloat16).transpose(1, 2)
    qsl_cpu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    qsl = qsl_cpu.to(dev)
    ci = torch.randperm(NCACHE, device=dev)[:len(lens)].to(torch.int32)
    hi = torch.tensor(init, device=dev, dtype=torch.bool)
    nd, bp, tco = compute_causal_conv1d_metadata(qsl_cpu, device=torch.device(dev))
    md = types.SimpleNamespace(nums_dict=nd, batch_ptr=bp, token_chunk_offset_ptr=tco)
    a = torch.randn(L, HV, device=dev, dtype=torch.bfloat16); b = torch.randn(L, HV, device=dev, dtype=torch.bfloat16)
    return x, cs, qsl, ci, hi, md, a, b


def stock(x, cs, qsl, ci, hi, md, a, b, n_dec):
    out = causal_conv1d_fn(x, w, bias, activation="silu", conv_states=cs, has_initial_state=hi, cache_indices=ci,
                           query_start_loc=qsl, metadata=md).transpose(0, 1)
    q, k, v, g, beta = fused_post_conv_prep(out[n_dec:], a[n_dec:], b[n_dec:], A_log, dt_bias, H, KD, VD)
    return out[:n_dec], q, k, v, g, beta


def fused(x, cs, qsl, ci, hi, md, a, b, n_dec):
    out, q, k, v = fconv.fused_conv_qkv(x, w, bias, cs, qsl, ci, hi, md, n_dec, H, KD)
    g, beta = fconv.gating(a[n_dec:], b[n_dec:], A_log, dt_bias)
    return out.transpose(0, 1)[:n_dec], q.view(-1, H, KD), k.view(-1, H, KD), v.view(-1, HV, VD), g, beta


def timeit(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); f(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


cases = [
    ("prefill 4x4096 no-init", [4096] * 4, [False] * 4, 0),
    ("prefill 16x1024 no-init", [1024] * 16, [False] * 16, 0),
    ("mixed 64dec+[2,5,3000,1000,517]", [1] * 64 + [2, 5, 3000, 1000, 517], [True] * 64 + [True, False, True, False, True], 64),
    ("mixed 200dec+1 chunk 2048", [1] * 200 + [2048], [True] * 200 + [True], 200),
]
allok = True
for dim_first in (True, False):
    for name, lens, init, n_dec in cases:
        x, cs, qsl, ci, hi, md, a, b = make_case(lens, init, dim_first)
        cs1, cs2 = cs.clone(), cs.clone()
        if not dim_first:  # keep the transposed layout for the clones
            cs1 = cs.transpose(1, 2).clone().transpose(1, 2); cs2 = cs.transpose(1, 2).clone().transpose(1, 2)
        r1 = stock(x, cs1, qsl, ci, hi, md, a, b, n_dec)
        r2 = fused(x, cs2, qsl, ci, hi, md, a, b, n_dec)
        names = ["dec_out", "q", "k", "v", "g", "beta"]
        diffs = {nm: (t1 != t2).sum().item() for nm, t1, t2 in zip(names, r1, r2)}
        diffs["conv_state"] = (cs1 != cs2).sum().item()
        maxd = max(((t1.float() - t2.float()).abs().max().item() if t1.numel() else 0.0) for t1, t2 in zip(r1, r2))
        ok = all(v == 0 for v in diffs.values())
        allok &= ok
        cs1b, cs2b = cs1.clone(), cs2.clone()
        ts = timeit(lambda: stock(x, cs1b, qsl, ci, hi, md, a, b, n_dec))
        tf = timeit(lambda: fused(x, cs2b, qsl, ci, hi, md, a, b, n_dec))
        print(f"{'DS' if dim_first else 'SD'} {name:34s} {'BITEXACT' if ok else 'DIFF'} {diffs} maxabs {maxd:.2e} | "
              f"stock {ts*1000:7.1f} us  fused {tf*1000:7.1f} us  ({(ts-tf)/ts*100:+.1f}%)", flush=True)
print("ALL BITEXACT" if allok else "SOME DIFF")
