"""gaps.py: GPU busy/idle split from an nsys sqlite (kernels + memcpy/memset), gap histogram, API launch counts."""
import sqlite3, sys
for p in sys.argv[1:]:
    c = sqlite3.connect(p)
    iv = []
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try: iv += c.execute(f"select start,end from {t}").fetchall()
        except sqlite3.OperationalError: pass
    iv.sort(); s0, e0 = iv[0][0], max(e for _, e in iv); busy = 0; cs, ce = iv[0]; gaps = []
    for s, e in iv[1:]:
        if s > ce: busy += ce - cs; gaps.append(s - ce); cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs; win = e0 - s0
    b = {"<0.1ms": 0, "0.1-1ms": 0, "1-5ms": 0, ">5ms": 0}; nb = dict.fromkeys(b, 0)
    for g in gaps:
        k = "<0.1ms" if g < 1e5 else "0.1-1ms" if g < 1e6 else "1-5ms" if g < 5e6 else ">5ms"; b[k] += g; nb[k] += 1
    print(f"{p.split('/')[-1]}: window {win/1e6:.0f} ms, GPU busy {100*busy/win:.1f}%, idle {(win-busy)/1e6:.0f} ms")
    for k in b: print(f"   gaps {k:8s} {b[k]/1e6:7.0f} ms  ({nb[k]} gaps)")
    q = """select s.value, count(*), sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id
           where s.value like '%Launch%' or s.value like '%Synchronize%' group by s.value order by 3 desc limit 6"""
    for n, k, t in c.execute(q): print(f"   api {n:40s} n={k:7d} {t/1e6:7.0f} ms")
