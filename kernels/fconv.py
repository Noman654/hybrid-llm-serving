# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM (https://github.com/vllm-project/vllm), files
#   model_executor/layers/mamba/ops/causal_conv1d.py (Copyright (c) 2024, Tri Dao; Copyright contributors to the vLLM project)
#   third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py (Copyright contributors to the vLLM project)
"""fconv.py: fused causal-conv1d + GDN post-conv prep for Qwen3.8 GDN prefill (vLLM 0.31).

Stock prefill does:  causal_conv1d_fn  -> out[L, 10240] (bf16)  -> fused_post_conv_prep reads out,
                     l2norm(q), l2norm(k), copies v, computes g/beta.
Fused:               one conv kernel whose epilogue l2-normalizes q/k per 128-wide head and writes
                     q/k/v straight into their final contiguous buffers; g/beta in a tiny kernel.
Saves one full write+read of the conv output. Math is kept identical to stock:
  - conv accumulation order, bf16 products and SiLU copied from _causal_conv1d_fwd_kernel
  - conv result rounded to bf16 BEFORE l2norm (stock stores bf16 then reloads)
Restrictions (else falls back to stock): kernel width 4, no prefix-cache (APC) args, no spec decode,
metadata provided, q/k region boundaries multiples of BLOCK_N=256 and head dim 128.
Decode rows of a mixed batch (first n_dec tokens, all length-1 sequences) are written raw to `out`
exactly like stock, because the decode path consumes them via rearrange_mixed_qkv.
Enable inside vLLM with:  import fconv; fconv.install()   (or env FCONV=1 via sitecustomize)
"""
import os
import torch
from vllm.triton_utils import tl, triton
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID


@triton.jit(do_not_specialize_on_alignment=["num_cache_lines"])
def _fconv_kernel(
    x_ptr, w_ptr, bias_ptr, conv_states_ptr, conv_state_indices_ptr, has_initial_states_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    o_ptr, q_ptr, k_ptr, v_ptr,
    n_dec,
    dim: tl.constexpr,
    num_cache_lines,
    stride_x_dim: tl.constexpr, stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr, stride_conv_state_dim: tl.constexpr, stride_conv_state_tok: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr, stride_o_token: tl.int64,
    pad_slot_id: tl.constexpr, null_block_id: tl.constexpr,
    HK: tl.constexpr,      # q (and k) region width = num_k_heads * head_k_dim
    HVV: tl.constexpr,     # v region width
    KD: tl.constexpr,      # head_k_dim (128)
    L2EPS: tl.constexpr,
    HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr, HAS_NULL_BLOCK: tl.constexpr,
    NP2_STATELEN: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    KERNEL_WIDTH: tl.constexpr = 4
    state_len = KERNEL_WIDTH - 1
    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    feat0 = tl.program_id(1) * BLOCK_N
    idx_feats = feat0 + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    conv_states_input_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa
        if conv_states_input_coord == null_block_id:
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            return
    conv_states_base = conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq) + (idx_feats * stride_conv_state_dim)
    w_base = w_ptr + (idx_feats * stride_w_dim)

    # ---- prologue: copied from _causal_conv1d_fwd_kernel (KERNEL_WIDTH == 4, APC disabled) ----
    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            col2 = tl.load(prior_tokens, mask_w, 0.0)
            col1 = tl.load(prior_tokens - 1 * stride_conv_state_tok, mask_w, 0.0)
            col0 = tl.load(prior_tokens - 2 * stride_conv_state_tok, mask_w, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tl.arange(0, NP2_STATELEN)
            x_ptrs = (x_ptr + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                      + (idx_feats * stride_x_dim)[None, :])
            mask_x = ((idx_tokens_last >= 0)[:, None] & (idx_tokens_last < seqlen)[:, None] & (idx_feats < dim)[None, :])
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)
            conv_states_output_coord = tl.load(conv_state_indices_ptr + idx_seq * stride_cache_indices).to(tl.int64)
            conv_states_ptrs_target = (conv_states_ptr + (conv_states_output_coord * stride_conv_state_seq)
                                       + (idx_feats * stride_conv_state_dim))[None, :] + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()
            tl.store(conv_states_ptrs_target, loaded_x, mask)
        else:
            if load_init_state:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                conv_states_ptrs_source = (conv_states_ptr + (conv_states_input_coord * stride_conv_state_seq)
                                           + (idx_feats * stride_conv_state_dim)[None, :]
                                           + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None])
                mask = ((conv_states_input_coord < num_cache_lines) & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :])
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)
                tl.debug_barrier()
                new_conv_state = tl.where(mask, conv_state, loaded_x)
                conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs = x_base[None, :] + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] & (idx_tokens_conv - VAL < seqlen)[:, None]
                          & (idx_feats < dim)[None, :])
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
                conv_states_ptrs_target = conv_states_base + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
    else:
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        col2 = tl.load(prior_tokens, mask_w, 0.0, cache_modifier=".ca")
        col1 = tl.load(prior_tokens - 1 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")
        col0 = tl.load(prior_tokens - 2 * stride_x_token, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + idx_feats, mask=idx_feats < dim, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token
    mask_w = idx_feats < dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # ---- epilogue routing (uniform per program) ----
    is_dec = sequence_start_index < n_dec
    is_qk = feat0 < 2 * HK
    row0 = sequence_start_index + token_offset - n_dec   # output row of token 0 in q/k/v
    if feat0 < HK:
        dst = q_ptr + feat0
    elif feat0 < 2 * HK:
        dst = k_ptr + (feat0 - HK)
    else:
        dst = v_ptr + (feat0 - 2 * HK)
    dst_stride = tl.where(is_qk, HK, HVV).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N)

    for idx_token in range(segment_len):
        acc = acc_preload
        # same order and dtypes as stock: acc(f32) += bf16(x*w)
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        matrix_x = tl.load(x_base_1d + idx_token * stride_x_token, mask=mask_x_1d)
        acc += matrix_x * w_col3
        col0 = col1
        col1 = col2
        col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        if is_dec:
            o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token) * stride_o_token + (idx_feats * stride_o_dim)
            tl.store(o_ptrs, acc, mask=mask_x_1d)
        else:
            y = acc.to(x_ptr.dtype.element_ty).to(tl.float32)   # stock rounds to bf16 before l2norm
            if is_qk:
                y2 = tl.reshape(y, (BLOCK_N // KD, KD))
                ss = _stock_order_sumsq(y2, BLOCK_N // KD)
                inv = 1.0 / tl.sqrt(ss + L2EPS)
                y = tl.reshape(y2 * inv[:, None], (BLOCK_N,))
            tl.store(dst + (row0 + idx_token) * dst_stride + offs_n, y.to(q_ptr.dtype.element_ty))


@triton.jit
def _stock_order_sumsq(y2, NH: tl.constexpr):
    """Sum of squares over 128 features in the same fp32 order as stock _fused_post_conv_kernel
    (layout sizePerThread [1,8], 16 lanes per row): each lane sums its 8 contiguous elements
    sequentially, then lanes combine by butterfly xor 8, 4, 2, 1."""
    sq = y2 * y2                                   # [NH, 128]
    s5 = tl.reshape(sq, (NH, 16, 2, 2, 2))         # element k = 4a + 2b + c inside the lane
    c0, c1 = tl.split(s5)                          # [NH,16,2,2] (a, b)
    c0b0, c0b1 = tl.split(c0)
    c1b0, c1b1 = tl.split(c1)
    e0, e4 = tl.split(c0b0)                        # (a=0,b=0,c=0)=0, (a=1,b=0,c=0)=4
    e2, e6 = tl.split(c0b1)
    e1, e5 = tl.split(c1b0)
    e3, e7 = tl.split(c1b1)
    p = e0 + e1
    p = p + e2
    p = p + e3
    p = p + e4
    p = p + e5
    p = p + e6
    p = p + e7                                     # [NH, 16] per-lane partials
    p = tl.reshape(p, (NH, 2, 8))                  # lane = 8*b3 + rest
    p = tl.sum(p, axis=1)                          # xor 8 (one add per pair: order-free)
    p = tl.reshape(p, (NH, 2, 4))
    p = tl.sum(p, axis=1)                          # xor 4
    p = tl.reshape(p, (NH, 2, 2))
    p = tl.sum(p, axis=1)                          # xor 2
    return tl.sum(p, axis=1)                       # xor 1


@triton.jit
def _gate_kernel(a_ptr, b_ptr, A_log_ptr, dt_bias_ptr, g_ptr, beta_ptr, stride_a_tok, stride_b_tok, L,
                 HV: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr, BLOCK_T: tl.constexpr):
    # verbatim gating branch of _fused_post_conv_kernel
    i_tb = tl.program_id(0)
    i_hv = tl.program_id(1)
    offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < L
    A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
    a_vals = tl.load(a_ptr + offs_t * stride_a_tok + i_hv, mask=mask_t, other=0).to(tl.float32)
    b_vals = tl.load(b_ptr + offs_t * stride_b_tok + i_hv, mask=mask_t, other=0).to(tl.float32)
    x = a_vals + dt_bias_val
    sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
    sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
    g_vals = -tl.exp(A_log_val) * sp
    beta_vals = tl.sigmoid(b_vals)
    tl.store(g_ptr + offs_t * HV + i_hv, g_vals, mask=mask_t)
    tl.store(beta_ptr + offs_t * HV + i_hv, beta_vals, mask=mask_t)


BLOCK_M, BLOCK_N = 8, 256


def fused_conv_qkv(x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state,
                   metadata, n_dec, num_k_heads, head_k_dim, activation="silu"):
    """x: (dim, L) channel-last view. Returns (out (dim, L) with decode columns valid, q, k, v)."""
    assert x.dtype == conv_states.dtype
    dim, L = x.shape
    HK = num_k_heads * head_k_dim
    HVV = dim - 2 * HK
    out = torch.empty_like(x)
    Lp = L - n_dec
    q = torch.empty(Lp, HK, dtype=x.dtype, device=x.device)
    k = torch.empty(Lp, HK, dtype=x.dtype, device=x.device)
    v = torch.empty(Lp, HVV, dtype=x.dtype, device=x.device)
    nd = metadata.nums_dict[BLOCK_M]
    tot = nd["tot"]
    batch_ptr, tco = nd["batch_ptr"], nd["token_chunk_offset_ptr"]
    grid = (tot, triton.cdiv(dim, BLOCK_N))
    _fconv_kernel[grid](
        x, weight, bias, conv_states, cache_indices, has_initial_state, query_start_loc, batch_ptr, tco,
        out, q, k, v, n_dec, dim, conv_states.size(0),
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        cache_indices.stride(0), out.stride(0), out.stride(1),
        PAD_SLOT_ID, NULL_BLOCK_ID,
        HK=HK, HVV=HVV, KD=head_k_dim, L2EPS=1e-6,
        HAS_BIAS=bias is not None, SILU_ACTIVATION=activation in ["silu", "swish", True],
        HAS_NULL_BLOCK=True, NP2_STATELEN=triton.next_power_of_2(weight.shape[1] - 1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_stages=2, num_warps=int(os.environ.get("FCONV_WARPS", "2")),
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out, q, k, v


def gating(a, b, A_log, dt_bias):
    L, HV = a.shape[0], A_log.shape[0]
    g = torch.empty(L, HV, dtype=torch.float32, device=a.device)
    beta = torch.empty(L, HV, dtype=torch.float32, device=a.device)
    if L:
        _gate_kernel[(triton.cdiv(L, 16), HV)](a, b, A_log, dt_bias, g, beta, a.stride(0), b.stride(0), L,
                                               HV=HV, SOFTPLUS_THRESHOLD=20.0, BLOCK_T=16, num_warps=4, num_stages=2)
    return g, beta


# ---------------- vLLM wiring ----------------
_stash = []
STATS = {"fused": 0, "fallback": 0}


def install():
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as M
    orig_conv, orig_prep = M.causal_conv1d_fn, M.fused_post_conv_prep

    def conv_wrap(x, weight, bias, activation=None, conv_states=None, has_initial_state=None, cache_indices=None,
                  query_start_loc=None, metadata=None, **kw):
        ok = (metadata is not None and not kw and weight.shape[1] == 4 and x.dtype == conv_states.dtype
              and getattr(metadata, "nums_dict", None) is not None and getattr(metadata, "num_spec_decodes", 0) == 0
              and x.stride(0) == 1 and x.shape[0] % BLOCK_N == 0 and cache_indices is not None and has_initial_state is not None)
        if ok:
            nkh, hk = _cfg.get(x.shape[0], (None, None))
            ok = nkh is not None and (nkh * hk) % BLOCK_N == 0 and hk == 128 and BLOCK_N % hk == 0
        if not ok:
            _stash.clear(); STATS["fallback"] += 1
            return orig_conv(x, weight, bias, activation=activation, conv_states=conv_states,
                             has_initial_state=has_initial_state, cache_indices=cache_indices,
                             query_start_loc=query_start_loc, metadata=metadata, **kw)
        n_dec = metadata.num_decode_tokens if metadata.num_decodes > 0 else 0
        out, q, k, v = fused_conv_qkv(x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state,
                                      metadata, n_dec, nkh, hk, activation)
        _stash.clear(); _stash.append((q, k, v)); STATS["fused"] += 1
        return out

    def prep_wrap(conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim, apply_l2norm=True,
                  output_g_exp=False):
        if _stash and apply_l2norm and not output_g_exp:
            q, k, v = _stash.pop()
            if q.shape[0] == conv_output.shape[0]:
                g, beta = gating(a, b, A_log, dt_bias)
                L = q.shape[0]
                return (q.view(L, num_k_heads, head_k_dim), k.view(L, num_k_heads, head_k_dim),
                        v.view(L, -1, head_v_dim), g, beta)
            raise RuntimeError(f"fconv stash length mismatch {q.shape[0]} vs {conv_output.shape[0]}")
        _stash.clear()
        return orig_prep(conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
                         apply_l2norm=apply_l2norm, output_g_exp=output_g_exp)

    M.causal_conv1d_fn = conv_wrap
    M.fused_post_conv_prep = prep_wrap
    print(f"[fconv] installed (conv dims cfg={_cfg})", flush=True)


# conv dim (per rank) -> (num_k_heads, head_k_dim); Qwen3.8-27B TP1: 2*16*128 + 48*128 = 10240
_cfg = {int(c.split(":")[0]): tuple(int(t) for t in c.split(":")[1].split("x"))
        for c in os.environ.get("FCONV_CFG", "10240:16x128").split(",")}
