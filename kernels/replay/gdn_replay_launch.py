"""gdn_replay_launch.py: launcher for the replay kernels (gdn_replay_kernels.py), lower host overhead for no-graph (mixed prefill+decode) steps.

Triton's JIT dispatch (binder + cache-key + hooks) costs ~15 us per launch on top of the driver launch. After the first
JIT launch for a given specialization we keep the CompiledKernel and launch it directly. The specialization key is
everything Triton specializes on that can vary between our calls: dtypes, 16-byte pointer alignment, int args (value,
since ==1 and %16 matter), and the constexpr/launch options. Falls back to JIT dispatch whenever a launch hook is set
(profilers) so tooling still sees launches.
"""
import torch
import triton
from triton import knobs
from gdn_replay_kernels import _pre_kernel, _main_kernel, _flush_kernel


def _hooked():
    for h in (knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook):
        if h is not None and (getattr(h, "calls", None) is None or h.calls):  # Triton >=3.6: empty HookChain by default
            return True
    return False


class FastLaunch:
    def __init__(self, fn):
        self.fn = fn
        self.const_names = [p.name for p in fn.params if p.is_constexpr]
        self.cache = {}

    def __call__(self, grid, args, consts, opts):
        key = []
        for a in args:
            if isinstance(a, torch.Tensor):
                key.append((a.dtype, a.data_ptr() & 15 == 0))
            elif isinstance(a, int):
                key.append(a)
            else:
                key.append(type(a))
        key = (tuple(key), tuple(consts.items()), tuple(opts.items()), grid[2] if len(grid) > 2 else 1)
        ck = self.cache.get(key)
        if ck is None or _hooked():
            ck = self.fn[grid](*args, **consts, **opts)
            self.cache[key] = ck
            return
        g = (grid[0], grid[1] if len(grid) > 1 else 1, grid[2] if len(grid) > 2 else 1)
        cvals = [consts[n] for n in self.const_names]
        ck.run(g[0], g[1], g[2], torch.cuda.current_stream().cuda_stream, ck.function, ck.packed_metadata, None,
               None, None, *args, *cvals)


_pre, _main, _flush = FastLaunch(_pre_kernel), FastLaunch(_main_kernel), FastLaunch(_flush_kernel)
_cd_cache = {}


def gdn_replay_decode7(mixed_qkv, a, b, A_log, dt_bias, scale, state, kr, ur, gr, out, idx, write_pos, is_flush,
                       BV=32, num_warps=1, num_stages=1, pre_warps=4, FBV=32, flush_warps=4, use_qk_l2norm_in_kernel=True,
                       cd=None, flush_rows=None, n_flush=None, max_flush=None, ring_ids=None):
    assert mixed_qkv.stride(-1) == 1 and a.stride(-1) == 1 and b.stride(-1) == 1
    if ring_ids is None:  # tests: ring indexed by state slot
        ring_ids = idx
    B = mixed_qkv.shape[0]
    HV, V, K = state.shape[-3:]
    H, L = kr.shape[1], kr.shape[2]
    LP = max(L, 16)
    if cd is None:
        key = (mixed_qkv.device, B, H, LP)
        if key not in _cd_cache:
            _cd_cache[key] = torch.empty(B, H, 2 * LP + 1, device=mixed_qkv.device, dtype=torch.float32)
        cd = _cd_cache[key]
    _pre((B, H), (mixed_qkv, kr, cd, idx, write_pos, ring_ids, mixed_qkv.stride(0), kr.stride(0)),
         dict(H=H, K=K, L=L, LP=LP, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel), dict(num_warps=pre_warps))
    _main((triton.cdiv(V, BV), B * HV),
          (mixed_qkv, a, b, A_log, dt_bias, out, state, kr, ur, gr, cd, idx, write_pos, ring_ids, is_flush, scale,
           mixed_qkv.stride(0), a.stride(0), b.stride(0), state.stride(0), kr.stride(0), ur.stride(0), gr.stride(0)),
          dict(H=H, HV=HV, K=K, V=V, BV=BV, L=L, LP=LP, SOFTPLUS_THRESHOLD=20.0, USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel),
          dict(num_warps=num_warps, num_stages=num_stages))
    if flush_rows is None:
        nz = torch.nonzero(is_flush).flatten().to(torch.int32)
        flush_rows = torch.zeros(B, dtype=torch.int32, device=state.device); flush_rows[: nz.numel()] = nz
        n_flush = torch.tensor([nz.numel()], dtype=torch.int32, device=state.device)
    MF = max_flush if max_flush is not None else B
    _flush((MF * HV,), (state, kr, ur, gr, idx, write_pos, ring_ids, flush_rows, n_flush, state.stride(0), kr.stride(0),
                        ur.stride(0), gr.stride(0)),
           dict(H=H, HV=HV, K=K, V=V, BV=FBV, L=L, LP=LP), dict(num_warps=flush_warps))
    return out
