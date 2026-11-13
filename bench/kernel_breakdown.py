"""kbreak.py SQLITE: GPU kernel time grouped into categories (top kernels listed)."""
import sqlite3, sys, re, collections
c = sqlite3.connect(sys.argv[1])
rows = c.execute("select s.value, sum(k.end-k.start), count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id group by s.value").fetchall()
tot = sum(r[1] for r in rows)
cat = collections.Counter()
def C(n):
    l = n.lower()
    for k, pat in [("solve_tril", r"solve_tril|merge_16x16|solve_|tril"), ("gdn_chunk", r"chunk|delta|wy|kkt|fwd_h|fwd_o|recompute|cumsum|gated|fullyfused|deltarule"),
                   ("conv1d", r"conv"), ("fp8_gemm", r"gemm|cutlass|sm90_xmma|deep_gemm|fp8|wgmma|nvjet"), ("attention", r"flash|attn|fmha"),
                   ("norm/act/quant", r"norm|silu|act|quant|rms|gelu|mul|add")]:
        if re.search(pat, l): return k
    return "other"
for n, t, k in rows: cat[C(n)] += t
print(f"total GPU kernel time {tot/1e6:.1f} ms")
for k, t in cat.most_common(): print(f"  {k:16s} {100*t/tot:5.1f}%  {t/1e6:8.1f} ms")
print("top kernels:")
for n, t, k in sorted(rows, key=lambda r: -r[1])[:18]: print(f"  {100*t/tot:5.1f}% n={k:6d} [{C(n)}] {n[:90]}")
