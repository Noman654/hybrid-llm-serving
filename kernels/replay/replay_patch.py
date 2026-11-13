"""replay_patch.py: runtime patch for vLLM 0.31 that runs Qwen3.8 GDN decode with the replay (deferred state write) kernel.
Import before building the LLM with GDN_REPLAY=1 (GDN_REPLAY_L = flush interval, default 8). History: tracks the per-request decode base itself, so vLLM's
use_replayssm mode is NOT needed (with CUDA graphs, that mode alone makes stock vLLM run-to-run nondeterministic).
NEEDS_RSSM = False tells the drivers not to set it.  Earlier doc (patch2): v2 of replay_patch.py: rings are a separate per-running-request buffer (ring id per request,
assigned on add_request, freed on remove_request), allocated right after the weights so the memory profiler counts it.
No change to the KV page size (v1 put the ring in the GDN state spec, which grew every pool page ~20% and cut
max concurrency). Original v1 doc: runtime patch that makes vLLM 0.31 (V1 model runner) run Qwen3.8 GDN decode with the
deferred-state-write kernel (gdn_replay5). Import before building the LLM; needs VLLM_USE_V2_MODEL_RUNNER=0,
VLLM_ENABLE_V1_MULTIPROCESSING=0 (in-process engine so the patch applies) and cache_config use_replayssm=True
(reuses vLLM's per-request decode-base bookkeeping that Mamba2 ReplaySSM uses).

Covers both decode paths: the packed decode-only step and decode rows inside mixed prefill+decode steps.
Ring buffers are 3 extra GDN state tensors (kv_cache[2:5]), so vLLM accounts and pages them like Mamba2 ReplaySSM.
Flush rules mirror vLLM's Mamba2 ReplaySSM (v1/attention/backends/mamba_attn.py): ring full, align-mode block
completion, or a query_len==1 prefill leftover.  GDN_REPLAY=0 (the default) disables everything (baseline).
"""
import os, sys, threading
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_replay_launch import gdn_replay_decode7 as gdn_replay_decode6

ON = os.environ.get("GDN_REPLAY", "0") == "1"
L = int(os.environ.get("GDN_REPLAY_L", "8"))
RING_DT = torch.bfloat16
DEBUG = os.environ.get("GDN_REPLAY_DEBUG") == "1"
NEEDS_RSSM = False
STATS = {"dbg_zero": 0, "dbg_dup": 0, "dbg_none": 0, "stash_restore": 0, "stash_miss": 0, "decode_steps": 0, "mixed_steps": 0, "mixed_cat": 0, "flush_rows": 0, "rows": 0}

from vllm.config.model import ModelConfig
from vllm.v1.attention.backends import gdn_attn as GA
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as Q
from vllm.v1.worker import gpu_model_runner as GMR

_ctx = threading.local()
_rings = {}  # layer prefix -> (kr, ur, gr)
_cd = {}     # device -> scratch


def _cfg(B):
    return dict(BV=32, num_warps=1, pre_warps=1, flush_warps=8) if B >= 128 else dict(BV=16, num_warps=1, pre_warps=1, flush_warps=4)


if ON:
    pass  # (patch4: vLLM ReplaySSM mode not used)

    # ---- metadata: write_pos / is_flush / compacted flush rows, persistent buffers (CUDA-graph safe) ----
    _orig_build = GA.GDNAttentionMetadataBuilder.build

    def build(self, common_prefix_len, common_attn_metadata, *args, **kw):
        m = common_attn_metadata
        md = _orig_build(self, common_prefix_len, m, *args, **kw)
        if not hasattr(self, "_rp"):
            mx = max(self.vllm_config.scheduler_config.max_num_seqs, self.decode_cudagraph_max_bs)
            d = self.device
            buf = torch.zeros(4 * mx + 1, dtype=torch.int32, device=d)  # [wp | is_flush | flush rows | n_flush | ring ids]
            self._rp = dict(buf=buf, wp=buf[:mx], fl=buf[mx: 2 * mx], fr=buf[2 * mx: 3 * mx], nf=buf[3 * mx: 3 * mx + 1],
                            rid=buf[3 * mx + 1:], mx=mx)
        rp = self._rp
        nd = md.num_decodes
        if nd == 0 or md.spec_sequence_masks is not None or md.num_spec_decodes:
            md.replay = None
            return md
        seq = m.seq_lens_cpu_upper_bound
        ib, rmap = _IB.get("batch"), _IB.get("rid")
        ids = ib.req_ids[:nd] if ib is not None else []
        bmap = _IB.get("base", {})
        if ids and any(r in bmap for r in ids):
            base = torch.tensor([bmap.get(r, 0) for r in ids] + [0] * (nd - len(ids)), dtype=torch.int64)
        else:
            base = None
        qsl = m.query_start_loc_cpu
        if base is None or seq is None:  # dummy/capture runs
            wp = torch.zeros(nd, dtype=torch.int64); fl = torch.zeros(nd, dtype=torch.bool)
        else:
            ql = qsl[1: nd + 1] - qsl[:nd]
            nc = seq[:nd].long() - ql.long()
            b0 = base[:nd].long()
            align = self.vllm_config.cache_config.mamba_cache_mode == "align"
            bs = self.kv_cache_spec.block_size
            eff = torch.maximum(b0, (nc // bs) * bs) if align else b0
            steps = nc - eff
            valid = ql > 0
            leftover = valid & (steps < 0)
            steps = torch.where(valid & ~leftover, steps, torch.zeros_like(steps))
            wp = torch.remainder(steps, L)
            fl = (wp == L - 1) | leftover
            if align:
                fl = fl | (valid & ((nc + ql) % bs == 0))
        nz = torch.nonzero(fl).flatten()
        mx = rp["mx"]
        host = torch.zeros(4 * mx + 1, dtype=torch.int32, pin_memory=True)  # fresh pinned tensor: the caching host
        host[:nd] = wp.to(torch.int32)                                       # allocator keeps it alive until the copy runs
        host[mx: mx + nd] = fl.to(torch.int32)
        host[2 * mx: 2 * mx + nz.numel()] = nz.to(torch.int32)
        host[3 * mx] = nz.numel()
        if rmap and base is not None:
            if ids:  # nd may include CUDA-graph padding rows beyond the live requests (they keep id 0)
                rl = [rmap.get(r, 0) for r in ids]
                host[3 * mx + 1: 3 * mx + 1 + len(ids)] = torch.tensor(rl, dtype=torch.int32)
                if DEBUG:
                    live = [x for r, x in zip(ids, rl) if r is not None]
                    STATS["dbg_zero"] += sum(1 for x in live if x == 0)
                    STATS["dbg_dup"] += len(live) - len(set(live))
                    STATS["dbg_none"] += sum(1 for r in ids if r is None)
        rp["buf"].copy_(host, non_blocking=True)
        STATS["flush_rows"] += int(nz.numel()); STATS["rows"] += nd
        md.replay = rp
        return md

    GA.GDNAttentionMetadataBuilder.build = build

    # ---- rings: separate buffers [R = max_num_seqs + 1, ...] per GDN layer, allocated after weights load ----
    from vllm.v1.worker import gpu_input_batch as GIB
    _IB = {}

    _orig_load = GMR.GPUModelRunner.load_model

    def load_model(self, *a, **k):
        r = _orig_load(self, *a, **k)
        R = self.vllm_config.scheduler_config.max_num_seqs + 1
        n = 0
        for m in self.model.modules():
            if isinstance(m, Q.QwenGatedDeltaNetAttention):
                H, HV = m.num_k_heads // m.tp_size, m.num_v_heads // m.tp_size
                dev = m.A_log.device
                _rings[m.prefix] = (torch.zeros(R, H, L, m.head_k_dim, dtype=RING_DT, device=dev),
                                    torch.zeros(R, HV, L, m.head_v_dim, dtype=RING_DT, device=dev),
                                    torch.zeros(R, HV, L, dtype=torch.float32, device=dev))
                n += 1
        if n:
            per = sum(t.numel() * t.element_size() for t in next(iter(_rings.values())))
            print(f"[gdn-replay] rings for {n} GDN layers x {R} request rows, {per * n / 2**30:.2f} GiB, L={L}", flush=True)
        _IB["free"] = list(range(R - 1, 0, -1)); _IB["rid"] = {}; _IB["base"] = {}
        return r

    GMR.GPUModelRunner.load_model = load_model

    _orig_ib_init = GIB.InputBatch.__init__

    def ib_init(self, *a, **k):
        _orig_ib_init(self, *a, **k)
        _IB["batch"] = self

    GIB.InputBatch.__init__ = ib_init
    _orig_add, _orig_rm = GIB.InputBatch.add_request, GIB.InputBatch.remove_request
    # V1 also removes running requests that are merely unscheduled this step and re-adds them later; their pending
    # (unflushed) ring must survive, so such removals are stashed and restored when the request comes back with
    # unchanged progress. Only scheduler-reported finished requests free their ring id. Preempted requests come back
    # with reset progress and get a fresh ring (their state is recomputed or restored from a flushed checkpoint).
    _IB["stash"] = {}; _IB["finishing"] = set()

    def add_request(self, request):
        rid = _IB.get("rid")
        st = _IB["stash"].pop(request.req_id, None)
        restore = st is not None and st[2] == request.num_computed_tokens
        if st is not None:
            STATS["stash_restore" if restore else "stash_miss"] += 1
            if not restore and DEBUG and len(_IB.setdefault("miss_log", [])) < 50:
                _IB["miss_log"].append((st[2], request.num_computed_tokens, len(request.prompt_token_ids or []), request.num_tokens if hasattr(request, "num_tokens") else None))
        if rid is not None and request.req_id not in rid:
            if restore:
                rid[request.req_id] = st[0]
            else:
                if st is not None:
                    _IB["free"].append(st[0])
                rid[request.req_id] = _IB["free"].pop()
        idx = _orig_add(self, request)
        # ring origin = full context at (re)admission (same rule as vLLM's ReplaySSM), or the stashed origin when an
        # unscheduled request comes back with unchanged progress (its pending ring steps are still valid)
        _IB["base"][request.req_id] = st[1] if restore else request.num_tokens
        return idx

    def remove_request(self, req_id):
        rid = _IB.get("rid")
        if rid is not None and req_id in rid:
            r = rid.pop(req_id)
            i = self.req_id_to_index.get(req_id)
            b0 = _IB["base"].pop(req_id, None)
            if req_id in _IB["finishing"] or i is None or b0 is None:
                _IB["free"].append(r)
            else:
                _IB["stash"][req_id] = (r, b0, int(self.num_computed_tokens_cpu[i]))
        return _orig_rm(self, req_id)

    _orig_upd = GMR.GPUModelRunner._update_states

    def _update_states(self, scheduler_output, *a, **k):
        fin = set(scheduler_output.finished_req_ids)
        _IB["finishing"] = fin
        for r in fin:  # finished while stashed (unscheduled): release
            st = _IB["stash"].pop(r, None)
            if st is not None:
                _IB["free"].append(st[0])
        try:
            return _orig_upd(self, scheduler_output, *a, **k)
        finally:
            _IB["finishing"] = set()

    GMR.GPUModelRunner._update_states = _update_states
    GIB.InputBatch.add_request, GIB.InputBatch.remove_request = add_request, remove_request

    def _scratch(dev, mx, H):
        key = (dev, mx, H)
        if key not in _cd:
            _cd[key] = torch.empty(mx, H, 2 * max(L, 16) + 1, dtype=torch.float32, device=dev)
        return _cd[key]

    def _run(layer, mixed_qkv, a, b, out, idx, rp, n):
        kr, ur, gr = _rings[layer.prefix]
        cd = _scratch(mixed_qkv.device, rp["mx"], kr.shape[1])
        gdn_replay_decode6(mixed_qkv, a, b, layer.A_log, layer.dt_bias, layer.head_k_dim ** -0.5, layer.kv_cache[1], kr, ur, gr,
                           out, idx, rp["wp"][:n], rp["fl"][:n], flush_rows=rp["fr"], n_flush=rp["nf"], max_flush=n, cd=cd[:n], ring_ids=rp["rid"][:n], **_cfg(n))

    # ---- decode-only step ----
    _orig_dec = Q.QwenGatedDeltaNetAttention._forward_core_decode_non_spec

    def _forward_core_decode_non_spec(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
        rp = getattr(attn_metadata, "replay", None)
        if rp is None or self.prefix not in _rings:
            return _orig_dec(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        idx_all = attn_metadata.non_spec_state_indices_tensor
        conv_state = self.kv_cache[0] if Q.is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)
        n = attn_metadata.num_actual_tokens
        mixed_qkv, b, a = mixed_qkv[:n], b[:n], a[:n]
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        x = Q.causal_conv1d_update(mixed_qkv, conv_state, conv_weights, self.conv1d.bias, self.activation,
                                   conv_state_indices=idx_all[:n], validate_data=False)
        if not torch.cuda.is_current_stream_capturing():
            STATS["decode_steps"] += 1
        _run(self, x, a, b, core_attn_out[:n].unsqueeze(1), idx_all[:n], rp, n)

    Q.QwenGatedDeltaNetAttention._forward_core_decode_non_spec = _forward_core_decode_non_spec

    # ---- mixed prefill+decode step: route the peeled decode rows through the replay kernel ----
    _orig_core = Q.QwenGatedDeltaNetAttention._forward_core

    def _forward_core(self, mixed_qkv, b, a, core_attn_out):
        md = Q.get_forward_context().attn_metadata
        md = md.get(self.prefix) if isinstance(md, dict) else None
        if md is not None and getattr(md, "replay", None) is not None and md.num_prefills > 0 and self.prefix in _rings:
            _ctx.layer, _ctx.md = self, md
            try:
                return _orig_core(self, mixed_qkv, b, a, core_attn_out)
            finally:
                _ctx.layer = None; _ctx.packed = None
        return _orig_core(self, mixed_qkv, b, a, core_attn_out)

    Q.QwenGatedDeltaNetAttention._forward_core = _forward_core

    _orig_fsg = Q.fused_sigmoid_gating_delta_rule_update

    def fused_sigmoid_gating_delta_rule_update(**kw):
        layer = getattr(_ctx, "layer", None)
        if layer is None:
            return _orig_fsg(**kw)
        md = _ctx.md
        n = md.num_decodes
        q, k, v = kw["q"], kw["k"], kw["v"]
        assert q.shape[1] == n and kw.get("num_accepted_tokens") is None and kw.get("use_qk_l2norm_in_kernel")
        packed = getattr(_ctx, "packed", None)
        _ctx.packed = None
        if packed is None:
            packed = torch.cat([q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)], dim=-1)
            STATS["mixed_cat"] += 1
        HV, V = v.shape[2], v.shape[3]
        out = torch.empty(n, 1, HV, V, dtype=q.dtype, device=q.device)
        STATS["mixed_steps"] += 1
        a_, b_ = kw["a"], kw["b"]
        if a_.stride(-1) != 1: a_ = a_.contiguous()
        if b_.stride(-1) != 1: b_ = b_.contiguous()
        _run(layer, packed, a_, b_, out, kw["ssm_state_indices"][:n], md.replay, n)
        return out.view(1, n, HV, V), None

    Q.fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update

    # ---- mixed step: the decode rows reach fsg via rearrange_mixed_qkv(packed rows), which cats q|k|v into a fresh
    # buffer. Our kernel reads the packed row layout directly, so keep the packed rows and return cheap views instead.
    _orig_rearr = Q.QwenGatedDeltaNetAttention.rearrange_mixed_qkv

    def rearrange_mixed_qkv(self, x):
        if x is not None and getattr(_ctx, "layer", None) is self and x.dim() == 2 and x.stride(-1) == 1 \
                and x.shape[0] == _ctx.md.num_decode_tokens:
            n = x.shape[0]
            qd = kd = self.key_dim // self.tp_size
            _ctx.packed = x
            return (x[:, :qd].view(1, n, -1, self.head_k_dim), x[:, qd: qd + kd].view(1, n, -1, self.head_k_dim),
                    x[:, qd + kd:].view(1, n, -1, self.head_v_dim))
        return _orig_rearr(self, x)

    Q.QwenGatedDeltaNetAttention.rearrange_mixed_qkv = rearrange_mixed_qkv
