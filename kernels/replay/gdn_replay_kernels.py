"""gdn_replay_kernels.py: Triton kernels for replay (deferred state write) GDN decode. Development history: v6 = v5 with rings addressed by a per-request ring id (ring_ids[n]) instead of the state slot, so the ring can be a small per-running-request buffer. v5 = v3 + state load issued first, unmasked ring loads sanitized by where (no wp->load latency chain), compacted flush rows (grid max_rows*HV, one full head per program). v3 = v2 with the flush moved to its own tensor-core kernel. v2 of the deferred-state-write GDN decode (math: see the docstring of _main_kernel below and the blog post).

Changes vs v1 (which lost to stock because the flush branch's tl.dot inflated registers for every
program and every program re-read the whole key ring):
  * precompute kernel, grid (B, H): normalizes k_t/q_t once per q/k head, dots them against the
    pending key ring (ck_j = k_j.k_t, cq_j = k_j.q_t, kq = k_t.q_t), stores k_t into ring row wp.
  * main kernel, grid (V/BV, B*HV): reads the anchor state once, the [L] dot vectors, G ring and its
    own u-ring tile; writes o, u_t, G_t. Flush rows rebuild the state with a runtime loop of rank-1
    updates (small register footprint) and write it back.
Ring row wp always receives the current step, so after a flush the ring is simply overwritten
from row 0 again (the next step must get write_pos 0).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pre_kernel(mixed_qkv, kr, cd, ssm_state_indices, write_pos, ring_ids, stride_mixed_qkv_tok, stride_kr_slot,
                H: tl.constexpr, K: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, USE_QK_L2NORM_IN_KERNEL: tl.constexpr):
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(ssm_state_indices + i_n).to(tl.int64)
    if slot <= 0:
        return
    wp = tl.load(write_pos + i_n)
    rslot = tl.load(ring_ids + i_n).to(tl.int64)
    o_k = tl.arange(0, K); o_l = tl.arange(0, LP)
    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    b_q = tl.load(p_mixed + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    p_kr = kr + rslot * stride_kr_slot + i_h * L * K
    m_l = o_l < wp
    R_k = tl.load(p_kr + o_l[:, None] * K + o_k[None, :], mask=(o_l < L)[:, None], other=0).to(tl.float32)
    R_k = tl.where(m_l[:, None], R_k, 0.0)
    p_cd = cd + (i_n * H + i_h) * (2 * LP + 1)
    tl.store(p_cd + o_l, tl.sum(R_k * b_k[None, :], 1))            # ck (unscaled)
    tl.store(p_cd + LP + o_l, tl.sum(R_k * b_q[None, :], 1))       # cq (q not yet scaled)
    tl.store(p_cd + 2 * LP, tl.sum(b_k * b_q))                     # kq
    tl.store(p_kr + wp * K + o_k, b_k.to(kr.dtype.element_ty))


@triton.jit
def _main_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, h, kr, ur, gr, cd,
    ssm_state_indices, write_pos, ring_ids, is_flush, scale,
    stride_mixed_qkv_tok, stride_a_tok, stride_b_tok,
    stride_h_slot, stride_kr_slot, stride_ur_slot, stride_gr_slot,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    o_l = tl.arange(0, LP)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    slot = tl.load(ssm_state_indices + i_n).to(tl.int64)
    if slot <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty))
        return
    p_h = h + slot * stride_h_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h).to(tl.float32)
    wp = tl.load(write_pos + i_n)
    rslot = tl.load(ring_ids + i_n).to(tl.int64)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    b_q = tl.load(p_mixed + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    x = a_val + tl.load(dt_bias + i_hv).to(tl.float32)
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * softplus_x
    beta_val = tl.sigmoid(b_val)

    p_ur = ur + rslot * stride_ur_slot + i_hv * L * V
    p_gr = gr + rslot * stride_gr_slot + i_hv * L
    p_cd = cd + (i_n * H + i_h) * (2 * LP + 1)
    # ring rows are loaded unconditionally (rows >= wp may hold stale or uninitialized data, so they are
    # zeroed with where, never multiplied by 0: that keeps NaN/inf garbage out)
    m_l = o_l < wp
    m_r = o_l < L
    R_G = tl.load(p_gr + o_l, mask=m_r, other=0.0)
    ck = tl.load(p_cd + o_l)
    cq = tl.load(p_cd + LP + o_l) * scale
    kq = tl.load(p_cd + 2 * LP) * scale
    R_u = tl.load(p_ur + o_l[:, None] * V + o_v[None, :], mask=m_r[:, None], other=0).to(tl.float32)  # [LP, BV]
    G_prev = tl.sum(tl.where(o_l == wp - 1, R_G, 0.0))
    G_t = G_prev + g_val
    w = tl.where(m_l, tl.exp(tl.where(m_l, G_t - R_G, 0.0)), 0.0)
    R_u = tl.where(m_l[:, None], R_u, 0.0)
    ck = tl.where(m_l, ck, 0.0); cq = tl.where(m_l, cq, 0.0)
    dG = tl.exp(G_t)
    u_t = beta_val * (b_v - dG * tl.sum(b_h * b_k[None, :], 1) - tl.sum((w * ck)[:, None] * R_u, 0))
    b_o = dG * tl.sum(b_h * b_q[None, :], 1) + tl.sum((w * cq)[:, None] * R_u, 0) + kq * u_t
    tl.store(p_o, b_o.to(p_o.dtype.element_ty))

    tl.store(p_ur + wp * V + o_v, u_t.to(ur.dtype.element_ty))
    if i_v == 0:
        tl.store(p_gr + wp, G_t)


@triton.jit
def _flush_kernel(h, kr, ur, gr, ssm_state_indices, write_pos, ring_ids, flush_rows, n_flush, stride_h_slot, stride_kr_slot, stride_ur_slot,
                  stride_gr_slot, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
                  L: tl.constexpr, LP: tl.constexpr):
    # S_t = e^{G_wp} S_a + sum_{j<=wp} e^{G_wp - G_j} u_j k_j^T   (ring row wp holds the current step)
    pid = tl.program_id(0)
    r, i_hv = pid // HV, pid % HV
    if r >= tl.load(n_flush):
        return
    i_n = tl.load(flush_rows + r)
    slot = tl.load(ssm_state_indices + i_n).to(tl.int64)
    if slot <= 0:
        return
    i_h = i_hv // (HV // H)
    wp = tl.load(write_pos + i_n)
    rslot = tl.load(ring_ids + i_n).to(tl.int64)
    o_k = tl.arange(0, K); o_l = tl.arange(0, LP)
    m_l = o_l <= wp
    p_gr = gr + rslot * stride_gr_slot + i_hv * L
    G_t = tl.load(p_gr + wp)
    w = tl.where(m_l, tl.exp(G_t - tl.load(p_gr + o_l, mask=m_l, other=0.0)), 0.0)
    R_k = tl.load(kr + rslot * stride_kr_slot + i_h * L * K + o_l[:, None] * K + o_k[None, :], mask=m_l[:, None], other=0).to(tl.float32)
    for i_v in range(V // BV):
        o_v = i_v * BV + tl.arange(0, BV)
        R_u = tl.load(ur + rslot * stride_ur_slot + i_hv * L * V + o_l[:, None] * V + o_v[None, :], mask=m_l[:, None], other=0).to(tl.float32)
        p_h = h + slot * stride_h_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h = tl.dot(tl.trans(w[:, None] * R_u), R_k, acc=tl.exp(G_t) * tl.load(p_h).to(tl.float32), input_precision="ieee")
        tl.store(p_h, b_h.to(p_h.dtype.element_ty))


_cd_cache = {}


def gdn_replay_decode6(mixed_qkv, a, b, A_log, dt_bias, scale, state, kr, ur, gr, out, idx, write_pos, is_flush,
                       BV=32, num_warps=1, num_stages=1, pre_warps=4, FBV=32, flush_warps=4, use_qk_l2norm_in_kernel=True, cd=None, flush_rows=None, n_flush=None, max_flush=None, ring_ids=None):
    if ring_ids is None:  # tests: ring indexed by state slot
        ring_ids = idx
    B = mixed_qkv.shape[0]
    HV, V, K = state.shape[-3:]
    H, L = kr.shape[1], kr.shape[2]
    LP = max(L, 16)
    if cd is None:  # scratch for the dot products; persistent so it is CUDA-graph safe
        key = (mixed_qkv.device, B, H, LP)
        if key not in _cd_cache:
            _cd_cache[key] = torch.empty(B, H, 2 * LP + 1, device=mixed_qkv.device, dtype=torch.float32)
        cd = _cd_cache[key]
    _pre_kernel[(B, H)](mixed_qkv, kr, cd, idx, write_pos, ring_ids, mixed_qkv.stride(0), kr.stride(0),
                        H=H, K=K, L=L, LP=LP, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, num_warps=pre_warps)
    _main_kernel[(triton.cdiv(V, BV), B * HV)](
        mixed_qkv, a, b, A_log, dt_bias, out, state, kr, ur, gr, cd, idx, write_pos, ring_ids, is_flush, scale,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), state.stride(0), kr.stride(0), ur.stride(0), gr.stride(0),
        H=H, HV=HV, K=K, V=V, BV=BV, L=L, LP=LP, SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps, num_stages=num_stages)
    if flush_rows is None:  # test convenience; in vLLM these come from the CPU-side metadata (fixed size for graphs)
        nz = torch.nonzero(is_flush).flatten().to(torch.int32)
        flush_rows = torch.zeros(B, dtype=torch.int32, device=state.device); flush_rows[: nz.numel()] = nz
        n_flush = torch.tensor([nz.numel()], dtype=torch.int32, device=state.device)
    MF = max_flush if max_flush is not None else B
    _flush_kernel[(MF * HV,)](state, kr, ur, gr, idx, write_pos, ring_ids, flush_rows, n_flush, state.stride(0), kr.stride(0),
        ur.stride(0), gr.stride(0), H=H, HV=HV, K=K, V=V, BV=FBV, L=L, LP=LP, num_warps=flush_warps)
    return out
