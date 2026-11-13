"""gen_figs.py: generates the blog's SVG figures. Every number here is a measured value quoted in hybrid-serving.md.
Static SVGs with light/dark styles (prefers-color-scheme works inside <img>-embedded SVG).
Palette: reference dataviz palette, blue (#2a78d6 / #3987e5) + orange highlight (#eb6834 / #d95926), validated."""
import os
OUT = os.path.dirname(os.path.abspath(__file__))
W = 720
FONT = "system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"

STYLE = """<style>
  .bg{fill:#fcfcfb} .t1{fill:#0b0b0b} .t2{fill:#52514e} .t3{fill:#7a7975}
  .grid{stroke:#e6e5e0;stroke-width:1} .axis{stroke:#c9c8c2;stroke-width:1}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .s0{fill:#d9d8d2} .s1l{stroke:#2a78d6} .s2l{stroke:#eb6834}
  .wash1{fill:#2a78d6;fill-opacity:.10} .box{fill:#f1f0ec;stroke:#d9d8d2}
  .gap{stroke:#fcfcfb;stroke-width:2}
  text{font-family:%s}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19} .t1{fill:#f0efec} .t2{fill:#c3c2b7} .t3{fill:#9a998f}
    .grid{stroke:#2f2f2d} .axis{stroke:#4a4a46}
    .s1{fill:#3987e5} .s2{fill:#d95926} .s0{fill:#3d3d3a} .s1l{stroke:#3987e5} .s2l{stroke:#d95926}
    .wash1{fill:#3987e5;fill-opacity:.14} .box{fill:#242423;stroke:#3d3d3a} .gap{stroke:#1a1a19}
  }
</style>""" % FONT


def svg(name, h, body, title, desc):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" '
         f'aria-labelledby="t d"><title id="t">{title}</title><desc id="d">{desc}</desc>{STYLE}'
         f'<rect class="bg" width="{W}" height="{h}" rx="8"/>{body}</svg>\n')
    open(os.path.join(OUT, name), "w").write(s)


def text(x, y, s, cls="t1", size=13, anchor="start", weight=400):
    return f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def hbar(x, y, w, h, cls, r=4):
    """horizontal bar, square at baseline (left), 4px rounded data end (right)."""
    if w <= r:
        return f'<rect x="{x}" y="{y}" width="{max(w,0.5)}" height="{h}" class="{cls}"/>'
    return (f'<path class="{cls}" d="M{x},{y} h{w-r} a{r},{r} 0 0 1 {r},{r} v{h-2*r} a{r},{r} 0 0 1 -{r},{r} h-{w-r} z"/>')


def title_block(t, sub):
    return text(24, 34, t, "t1", 16, weight=600) + text(24, 56, sub, "t2", 13)


# ---------- Fig 1: what a decode step moves ----------
def fig1():
    # 256 requests, bf16 state: weights ~30 GB read once; GDN state 19 GB read + 19 GB written; H100 3.35 TB/s
    rows = [("Weights (read)", 29.7, "s1", "≈ 9 ms"), ("GDN state (read)", 19.0, "s2", ""), ("GDN state (write)", 19.0, "s2", "")]
    x0, sc, y = 190, 11.0, 92
    b = title_block("What one decode step moves through memory", "256 concurrent requests, bf16 GDN state, H100 at 3.35 TB/s")
    for g in range(0, 41, 10):
        gx = x0 + g * sc
        b += f'<line x1="{gx}" y1="76" x2="{gx}" y2="{y+3*40-6}" class="grid"/>' + text(gx, y + 3 * 40 + 10, f"{g} GB", "t3", 11, "middle")
    for i, (lab, gb, c, note) in enumerate(rows):
        yy = y + i * 40
        b += text(x0 - 12, yy + 17, lab, "t2", 13, "end") + hbar(x0, yy, gb * sc, 24, c) + text(x0 + gb * sc + 8, yy + 17, f"{gb:.0f} GB", "t1", 13)
    # bracket for state
    bx = x0 + 19 * sc + 62
    b += f'<path d="M{bx},{y+44} h8 v64 h-8" fill="none" class="axis" stroke-width="1.5"/>'
    b += text(bx + 16, y + 74, "38 GB", "t1", 13, weight=600) + text(bx + 16, y + 92, "≈ 11.5 ms", "t2", 12)
    b += text(x0 + 29.7 * sc + 60, y + 17, "≈ 9 ms", "t2", 12)
    b += text(24, 262, "The state traffic is larger than the weight traffic, so the state's size and dtype matter a lot.", "t2", 12)
    svg("fig1-decode-bytes.svg", 280, b, "What one decode step moves",
        "Bar chart: weights 30 GB read; GDN state 19 GB read and 19 GB written per decode step at 256 requests.")


# ---------- Fig 2: replay mechanics ----------
def fig2():
    b = title_block("Writing the GDN state less often", "Replay writes the full state every L steps and keeps small per-step inputs in a ring buffer.")
    steps = 9
    x0, cw = 150, 58
    # row labels
    b += text(24, 118, "Stock", "t1", 13, weight=600) + text(24, 134, "full write each step", "t3", 11)
    b += text(24, 218, "Replay (L = 8)", "t1", 13, weight=600) + text(24, 234, "full write every 8th", "t3", 11)
    for i in range(steps):
        cx = x0 + i * cw
        b += text(cx + 22, 88, f"t{i+1}", "t3", 11, "middle")
        # stock: big block each step
        b += f'<rect x="{cx}" y="100" width="44" height="40" rx="4" class="s2"/>'
        # replay: big write only at step 8; else small ring entry
        if i == 7:
            b += f'<rect x="{cx}" y="200" width="44" height="40" rx="4" class="s2"/>'
        else:
            b += f'<rect x="{cx+14}" y="226" width="16" height="14" rx="3" class="s1"/>'
    b += f'<line x1="{x0-6}" y1="252" x2="{x0+steps*cw}" y2="252" class="axis"/>'
    # legend
    ly = 286
    b += f'<rect x="150" y="{ly-11}" width="14" height="14" rx="3" class="s2"/>' + text(170, ly, "full state write (75 MB per request, bf16)", "t2", 12)
    b += f'<rect x="440" y="{ly-8}" width="10" height="9" rx="2" class="s1"/>' + text(458, ly, "ring entry: k, corrected v, decay", "t2", 12)
    b += text(24, 318, "To read the current state, the kernel rebuilds it from the last full write plus the ring entries since then.", "t2", 12)
    b += text(24, 336, "At 256 requests: ~17 GB of writes skipped per step, ~1.6 GB of ring reads added.", "t2", 12)
    b += text(24, 354, "Measured decode step: 36.2 → 32.2 ms.", "t2", 12)
    svg("fig2-replay.svg", 374, b, "Replay: writing the state every L steps",
        "Diagram: stock writes a full state block every step; replay writes a full block only at step 8 and small ring entries in between.")


# ---------- Fig 3: step sizes vs capture limit ----------
def fig3():
    b = title_block("Which steps get a CUDA graph", "Tokens per engine step in the serving mix (512-token prompts, 256 concurrent)")
    x0, x1, vmax = 190, 690, 2200
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    rows = [("Default limit (512)", 512, 92), ("Raised limit (2,048)", 2048, 128)]
    for lab, lim, yy in rows:
        b += text(x0 - 12, yy + 16, lab, "t2", 13, "end")
        b += f'<rect x="{x0}" y="{yy}" width="{sx(lim)-x0}" height="22" class="wash1"/>'
        b += text(sx(lim) - 8, yy + 15, "graphs", "t2", 11, "end")
    y = 186
    b += text(x0 - 12, y + 4, "Step sizes", "t2", 13, "end")
    b += f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="axis"/>'
    for v in (0, 512, 1024, 1536, 2048):
        b += f'<line x1="{sx(v)}" y1="82" x2="{sx(v)}" y2="{y+5}" class="grid"/>' + text(sx(v), y + 48, f"{v:,}", "t3", 11, "middle")
    for v, c, lab in ((256, "s1", "decode-only ≤ 256"), (770, "s2", "median mixed ≈ 770"), (1300, "s2", "p90 ≈ 1,300")):
        b += f'<circle cx="{sx(v)}" cy="{y}" r="8" class="bg"/><circle cx="{sx(v)}" cy="{y}" r="6" class="{c}"/>'
        b += text(sx(v), y - 14 if v == 770 else y + 26, lab, "t1", 12, "middle")
    b += text(24, 270, "46% of wall time was in steps between 512 and 2,048 tokens. At the default limit none of them got a graph,", "t2", 12)
    b += text(24, 288, "so Python launched each step's kernels one by one while the GPU waited.", "t2", 12)
    svg("fig3-step-sizes.svg", 306, b, "Step sizes vs the CUDA-graph limit",
        "Number line of tokens per step: decode-only steps up to 256, median mixed step 770, p90 1300; graphs cover up to 512 by default and up to 2048 when raised.")


# ---------- Fig 4: GPU busy before/after ----------
def fig4():
    b = title_block("GPU busy time in steady-state serving", "nsys, 1 H100, serving mix; share of wall time with a kernel running")
    x0, wbar, y = 190, 440, 90
    rows = [("Default graph limit (512)", 67), ("Graph limit 2,048", 87)]
    for i, (lab, busy) in enumerate(rows):
        yy = y + i * 54
        bw = wbar * busy / 100
        b += text(x0 - 12, yy + 18, lab, "t2", 13, "end")
        b += f'<rect x="{x0}" y="{yy}" width="{bw}" height="26" class="s1"/>'
        b += f'<rect x="{x0+bw+2}" y="{yy}" width="{wbar-bw-2}" height="26" class="s0"/>'
        b += text(x0 + 10, yy + 18, f"busy {busy}%", "t1", 12, weight=600).replace('class="t1"', 'fill="#ffffff"')
        b += text(x0 + wbar + 10, yy + 18, f"idle {100-busy}%", "t2", 12)
    b += text(24, 226, "What idle time remains: 5–6% between steps (scheduler work), 2–3% inside steps, the rest at kernel boundaries.", "t2", 12)
    svg("fig4-gpu-busy.svg", 246, b, "GPU busy time before and after",
        "Two bars: GPU busy 67% at the default graph limit, 87% with a 2048 limit.")


# ---------- Fig 5: the ladder ----------
def fig5():
    rows = [("Stock vLLM", 210.1, "1.00×"), ("+ bf16 GDN state", 171.3, "1.23×"), ("+ prefix cache off", 144.6, "1.45×"),
            ("+ graph limit 2,048", 128.8, "1.63×"), ("+ replay kernel", 122.5, "1.72×")]
    b = title_block("One benchmark, one change at a time", "Time to serve 1,024 requests on one H100 (lower is better)")
    x0, sc, y = 190, 1.95, 84
    for g in range(0, 221, 50):
        gx = x0 + g * sc
        b += f'<line x1="{gx}" y1="{y-8}" x2="{gx}" y2="{y+5*38-6}" class="grid"/>' + text(gx, y + 5 * 38 + 10, f"{g} s", "t3", 11, "middle")
    for i, (lab, t, sp) in enumerate(rows):
        yy = y + i * 38
        c = "s2" if i == 3 else "s1"
        b += text(x0 - 12, yy + 16, lab, "t2", 13, "end") + hbar(x0, yy, t * sc, 22, c)
        b += text(x0 + t * sc + 8, yy + 16, f"{t:.1f} s", "t1", 13) + text(x0 + t * sc + 66, yy + 16, sp, "t3", 12)
    b += text(24, 306, "Highlighted: the one config value that also held up on a tuned 8-GPU node (+10.4%).", "t2", 12)
    svg("fig5-ladder.svg", 324, b, "The optimization ladder",
        "Bar chart of total time: stock 210.1 s, bf16 state 171.3 s, prefix cache off 144.6 s, graph limit 2048 128.8 s, replay kernel 122.5 s.")


# ---------- Fig 6: prefill kernel shares ----------
def fig6():
    rows = [("FP8 matrix multiplies", 71.1), ("Norms, activations, quant.", 17.4), ("Other", 5.0),
            ("GDN conv1d + prep", 3.5), ("GDN core (chunked)", 2.8), ("Attention", 0.2)]
    b = title_block("Where prefill GPU time goes", "Share of kernel time, 256 prompts × 4,096 tokens, one H100 (FlashInfer GDN prefill)")
    x0, sc, y = 210, 5.6, 84
    for g in range(0, 81, 20):
        gx = x0 + g * sc
        b += f'<line x1="{gx}" y1="{y-8}" x2="{gx}" y2="{y+6*34-6}" class="grid"/>' + text(gx, y + 6 * 34 + 10, f"{g}%", "t3", 11, "middle")
    for i, (lab, p) in enumerate(rows):
        yy = y + i * 34
        c = "s2" if "conv1d" in lab else "s1"
        b += text(x0 - 12, yy + 15, lab, "t2", 13, "end") + hbar(x0, yy, p * sc, 20, c) + text(x0 + p * sc + 8, yy + 15, f"{p}%", "t1", 13)
    b += text(24, 320, "We made the highlighted kernels 30% faster, bit-exactly. End-to-end prefill: +0.8%, as their 3.5% share predicts.", "t2", 12)
    svg("fig6-prefill-shares.svg", 338, b, "Where prefill GPU time goes",
        "Bar chart of prefill kernel time shares: FP8 GEMM 71.1%, norms/activations/quant 17.4%, other 5%, conv1d+prep 3.5%, GDN core 2.8%, attention 0.2%.")


for f in (fig1, fig2, fig3, fig4, fig5, fig6):
    f()
print("ok", sorted(x for x in os.listdir(OUT) if x.endswith(".svg")))
