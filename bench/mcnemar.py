"""mcnemar.py A.json B.json ...: per-question GSM8K agreement and exact McNemar vs the first file."""
import json, re, sys, math, glob
import pandas as pd
from huggingface_hub import hf_hub_download
te = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test-00000-of-00001.parquet", repo_type="dataset"))
def ans(t):
    m = re.findall(r"####\s*(-?[\d,\.]+)", t) or re.findall(r"-?\d[\d,]*\.?\d*", t)
    return m[-1].replace(",", "").rstrip(".") if m else None
F = sys.argv[1:]; R = [json.load(open(f))["res"] for f in F]; N = len(R[0])
gold = [a.split("####")[-1].strip().replace(",", "") for a in te.answer[:N]]
C = [[ans(r["text"]) == g for r, g in zip(res, gold)] for res in R]
for f, c, res in zip(F, C, R):
    lost = sum(a and not b for a, b in zip(C[0], c)); won = sum(b and not a for a, b in zip(C[0], c)); k = lost + won
    p = min(1, 2 * sum(math.comb(k, t) for t in range(min(lost, won) + 1)) / 2 ** k) if k else 1
    same = sum(x["toks"] == y["toks"] for x, y in zip(R[0], res))
    print(f"{f.split('/')[-1]:20s} acc {sum(c)/N:.4f}  vs first: lost {lost} won {won} McNemar p={p:.3f}  identical outputs {same}/{N}")
