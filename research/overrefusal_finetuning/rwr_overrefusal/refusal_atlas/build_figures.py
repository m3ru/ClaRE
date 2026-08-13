#!/usr/bin/env python3
"""Emit the shareable Refusal Atlas figure page (self-contained, theme-aware HTML).

Reads results/figures_data.json (all metrics on the independent Sonnet-judge ground
truth) and writes figures.html: four hand-authored inline-SVG figures + narrative.
Geometry is computed from the data so the marks are exact.
"""
import json, html, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "results", "figures_data.json")))
LL, QW = D["models"]["llama"], D["models"]["qwen"]

# ------------------------------------------------------------------ helpers
def esc(s): return html.escape(str(s))

def lerp(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)

# ---- FIG 1: topic over-refusal, paired dot plot ------------------
def fig_topics():
    order = sorted(LL["topic_rate"], key=lambda k: -LL["topic_rate"][k])
    W, rh, top, bot = 900, 33, 34, 44
    x0, x1 = 176, 660          # plot area for [0,1]
    cL, cQ = 748, 828          # value columns
    H = top + rh * len(order) + bot
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Over-refusal rate by topic, per model" style="height:auto">']
    # gridlines + axis ticks
    for t in (0, .25, .5, .75, 1.0):
        x = lerp(t, 0, 1, x0, x1)
        p.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{top+rh*len(order)}" class="grid"/>')
        p.append(f'<text x="{x:.1f}" y="{top+rh*len(order)+18}" class="tick" text-anchor="middle">{int(t*100)}%</text>')
    # column headers
    p.append(f'<text x="{x0}" y="{top-16}" class="eyebrow-svg" text-anchor="start">OR-BENCH-HARD TOPIC</text>')
    p.append(f'<text x="{cL}" y="{top-16}" class="col-hd llama-t" text-anchor="middle">Llama</text>')
    p.append(f'<text x="{cQ}" y="{top-16}" class="col-hd qwen-t" text-anchor="middle">Qwen</text>')
    for i, k in enumerate(order):
        y = top + rh * i + rh * 0.5
        vl, vq = LL["topic_rate"][k], QW["topic_rate"][k]
        xl, xq = lerp(vl, 0, 1, x0, x1), lerp(vq, 0, 1, x0, x1)
        p.append(f'<text x="{x0-14}" y="{y+4}" class="rowlab" text-anchor="end">{esc(k)}</text>')
        # connector = the model gap
        p.append(f'<line x1="{xq:.1f}" y1="{y:.1f}" x2="{xl:.1f}" y2="{y:.1f}" class="conn"/>')
        p.append(f'<circle cx="{xq:.1f}" cy="{y:.1f}" r="6.5" class="dot qwen-f"/>')
        p.append(f'<circle cx="{xl:.1f}" cy="{y:.1f}" r="6.5" class="dot llama-f"/>')
        p.append(f'<text x="{cL}" y="{y+4}" class="val llama-t" text-anchor="middle">{vl*100:.0f}</text>')
        p.append(f'<text x="{cQ}" y="{y+4}" class="val qwen-t" text-anchor="middle">{vq*100:.0f}</text>')
    p.append('</svg>')
    return "\n".join(p)

# ---- FIG 2: full vs within-topic AUC dumbbell (two stacked facets)
def fig_fidelity():
    sigs = [("vector", "refusal direction"), ("probe", "probe ensemble"), ("logit_sum", "output logit")]
    lo, hi = 0.85, 1.00
    W, rowh, ptop, pgap = 900, 42, 46, 34
    a0, a1 = 214, 700          # axis
    vx, dx = 762, 842          # within-value column, drop column
    panels = [("Llama", LL, "llama"), ("Qwen", QW, "qwen")]
    ph = ptop + rowh * len(sigs) + 28
    H = ph * 2 + pgap
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Full vs within-topic AUC per signal" style="height:auto">']
    for pi, (name, M, cls) in enumerate(panels):
        oy = pi * (ph + pgap)
        p.append(f'<text x="60" y="{oy+ptop-22}" class="col-hd {cls}-t" text-anchor="start">{name}</text>')
        p.append(f'<text x="{vx}" y="{oy+ptop-22}" class="eyebrow-svg" text-anchor="middle">WITHIN</text>')
        p.append(f'<text x="{dx}" y="{oy+ptop-22}" class="eyebrow-svg" text-anchor="middle">DROP</text>')
        ybot = oy + ptop + rowh * len(sigs) - 12
        for t in (0.85, 0.90, 0.95, 1.00):
            x = lerp(t, lo, hi, a0, a1)
            p.append(f'<line x1="{x:.1f}" y1="{oy+ptop-10}" x2="{x:.1f}" y2="{ybot}" class="grid"/>')
            p.append(f'<text x="{x:.1f}" y="{ybot+18}" class="tick" text-anchor="middle">{t:.2f}</text>')
        for si, (key, lab) in enumerate(sigs):
            y = oy + ptop + rowh * si + rowh * 0.4
            full, within = M["auc"][key]["full"], M["auc"][key]["within"]
            xf = lerp(full, lo, hi, a0, a1)
            xw = lerp(max(within, lo), lo, hi, a0, a1)
            p.append(f'<text x="60" y="{y+4:.1f}" class="rowlab-sm" text-anchor="start">{esc(lab)}</text>')
            p.append(f'<line x1="{xf:.1f}" y1="{y:.1f}" x2="{xw:.1f}" y2="{y:.1f}" class="conn2"/>')
            p.append(f'<circle cx="{xf:.1f}" cy="{y:.1f}" r="5" class="dot-hollow"/>')
            p.append(f'<circle cx="{xw:.1f}" cy="{y:.1f}" r="6" class="dot {cls}-f"/>')
            drop = full - within
            dtxt = f'–{drop*100:.1f}' if drop > 0.005 else '±0'
            dcls = "warn" if drop > 0.03 else "muted-t"
            p.append(f'<text x="{vx}" y="{y+4:.1f}" class="val {cls}-t" text-anchor="middle">{within:.3f}</text>')
            p.append(f'<text x="{dx}" y="{y+4:.1f}" class="drop {dcls}" text-anchor="middle">{dtxt}</text>')
    p.append('</svg>')
    return "\n".join(p)

# ---- FIG 3: circularity, AUC inflation from the regex label ------
def fig_circularity():
    sigs = [("vector", "refusal direction"), ("probe", "probe ensemble"), ("logit_sum", "output logit")]
    def infl(M, k): return M["auc"][k]["regex"] - M["auc"][k]["judge"]
    hi = 0.025
    W, rh, top = 900, 40, 46
    gap = 15
    x0, x1 = 210, 560          # bar area
    H = top + (rh) * len(sigs) + 40
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="AUC inflation from the circular regex label" style="height:auto">']
    for t in (0, 0.005, 0.01, 0.015, 0.02, 0.025):
        x = lerp(t, 0, hi, x0, x1)
        p.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{top+rh*len(sigs)-8}" class="grid"/>')
        p.append(f'<text x="{x:.1f}" y="{top+rh*len(sigs)+4}" class="tick" text-anchor="middle">{t*100:.1f}</text>')
    p.append(f'<text x="{x1}" y="{top+rh*len(sigs)+20}" class="tick" text-anchor="end">AUC points overstated (×100)</text>')
    bh = 11
    for si, (key, lab) in enumerate(sigs):
        yc = top + rh * si + rh * 0.4
        p.append(f'<text x="{x0-16}" y="{yc+4}" class="rowlab-sm" text-anchor="end">{esc(lab)}</text>')
        for (M, cls, off) in ((LL, "llama", -bh-1), (QW, "qwen", 1)):
            v = infl(M, key)
            w = lerp(v, 0, hi, x0, x1) - x0
            y = yc + off
            p.append(f'<rect x="{x0}" y="{y:.1f}" width="{max(w,0.5):.1f}" height="{bh}" rx="3" class="{cls}-fill"/>')
            p.append(f'<text x="{x0+w+7:.1f}" y="{y+bh-2:.1f}" class="barval {cls}-t">{v*100:.1f}</text>')
    p.append('</svg>')
    return "\n".join(p)

# ---- FIG 4: single-word triggers, two panels --------------------
def fig_triggers():
    shared = {"attack", "contraband", "exploit", "lethal", "weaponize"}
    hi = 0.45
    rowh, top = 34, 50
    n = 6
    def panel(M, cls, name, ox):
        words = M["words"][:n]
        x0, x1 = ox + 118, ox + 360
        out = [f'<text x="{ox+10}" y="{top-22}" class="col-hd {cls}-t">{name}</text>']
        for t in (0, .15, .30, .45):
            x = lerp(t, 0, hi, x0, x1)
            out.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{top+rowh*n-10}" class="grid"/>')
            out.append(f'<text x="{x:.1f}" y="{top+rowh*n+6}" class="tick" text-anchor="middle">+{int(t*100)}</text>')
        for i, (w, d, cnt) in enumerate(words):
            y = top + rowh * i + 6
            bw = lerp(d, 0, hi, x0, x1) - x0
            sh = w in shared
            out.append(f'<text x="{x0-12}" y="{y+11}" class="rowlab-sm {"shared" if sh else ""}" text-anchor="end">{esc(w)}{" ✦" if sh else ""}</text>')
            out.append(f'<rect x="{x0}" y="{y}" width="{max(bw,0.5):.1f}" height="15" rx="3.5" class="{cls}-fill"/>')
            out.append(f'<text x="{x0+bw+7:.1f}" y="{y+12}" class="barval {cls}-t">+{d*100:.0f}</text>')
        return "\n".join(out)
    W = 900
    H = top + rowh * n + 34
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Single-word triggers per model" style="height:auto">']
    p.append(panel(LL, "llama", "Llama", 0))
    p.append(panel(QW, "qwen", "Qwen", 470))
    p.append('</svg>')
    return "\n".join(p)

# ------------------------------------------------------------------ source stat tiles
def stat_row():
    rows = [("orbench_toxic", "Toxic anchor", "should refuse"),
            ("orbench_hard", "OR-Bench-Hard", "benign, sensitive-topic"),
            ("xstest", "XSTest", "benign, safe-looking-unsafe"),
            ("single_edit_pair", "Minimal pairs", "benign controls")]
    tiles = []
    for key, name, sub in rows:
        vl, vq = LL["source_rate"][key], QW["source_rate"][key]
        tiles.append(f'''<div class="tile">
  <div class="tile-name">{esc(name)}</div>
  <div class="tile-sub">{esc(sub)}</div>
  <div class="tile-vals">
    <span class="tv llama-t">{vl*100:.0f}<span class="pct">%</span></span>
    <span class="tv qwen-t">{vq*100:.0f}<span class="pct">%</span></span>
  </div>
</div>''')
    return '<div class="tiles">' + "\n".join(tiles) + '</div>'

# data table (accessibility / table view)
def data_table():
    order = sorted(LL["topic_rate"], key=lambda k: -LL["topic_rate"][k])
    rows = "".join(
        f'<tr><td>{esc(k)}</td><td class="num">{LL["topic_rate"][k]*100:.0f}%</td>'
        f'<td class="num">{QW["topic_rate"][k]*100:.0f}%</td></tr>' for k in order)
    return f'''<details class="tablewrap"><summary>Topic table (all values)</summary>
<table><thead><tr><th>Topic</th><th class="num">Llama</th><th class="num">Qwen</th></tr></thead>
<tbody>{rows}</tbody></table></details>'''

# ------------------------------------------------------------------ assemble
CSS = """
:root{
  color-scheme:light;
  --page:#f5f6f8; --surface:#ffffff; --surface-2:#f8f9fb;
  --ink:#111418; --ink-2:#565d67; --muted:#8a919c;
  --grid:#e6e8ec; --hair:#e0e3e8; --hair-strong:#d2d6dd;
  --llama:#eb6834; --qwen:#2a78d6;
  --llama-soft:rgba(235,104,52,.12); --qwen-soft:rgba(42,120,214,.12);
  --warn:#c2410c;
  --accent:#2a78d6;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;
  --page:#0c0e11; --surface:#15181d; --surface-2:#191d23;
  --ink:#f1f3f6; --ink-2:#aab1bd; --muted:#727a86;
  --grid:#262a31; --hair:#242830; --hair-strong:#333842;
  --llama:#e8703f; --qwen:#4a92ec;
  --llama-soft:rgba(232,112,63,.16); --qwen-soft:rgba(74,146,236,.16);
  --warn:#f0863f;
  --accent:#4a92ec;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0c0e11; --surface:#15181d; --surface-2:#191d23;
  --ink:#f1f3f6; --ink-2:#aab1bd; --muted:#727a86;
  --grid:#262a31; --hair:#242830; --hair-strong:#333842;
  --llama:#e8703f; --qwen:#4a92ec;
  --llama-soft:rgba(232,112,63,.16); --qwen-soft:rgba(74,146,236,.16);
  --warn:#f0863f;
  --accent:#4a92ec;
}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.62;-webkit-font-smoothing:antialiased;}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Code","JetBrains Mono",Menlo,Consolas,monospace;}
.wrap{max-width:940px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(18px,4vw,40px) 96px;}
.eyebrow{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:12px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--accent);font-weight:600;margin:0 0 14px;}
h1{font-size:clamp(30px,5vw,46px);line-height:1.06;letter-spacing:-.02em;font-weight:700;
  margin:0 0 18px;text-wrap:balance;}
.lede{font-size:clamp(17px,2.2vw,20px);color:var(--ink-2);max-width:66ch;margin:0 0 8px;text-wrap:pretty;}
.meta{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--muted);margin-top:22px;
  display:flex;flex-wrap:wrap;gap:8px 22px;border-top:1px solid var(--hair);padding-top:18px;}
.meta b{color:var(--ink-2);font-weight:600;}
section{margin-top:60px;}
h2{font-size:clamp(21px,3vw,27px);line-height:1.2;letter-spacing:-.015em;font-weight:700;margin:0 0 6px;text-wrap:balance;}
.kicker{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--muted);font-weight:600;margin:0 0 10px;}
p{max-width:68ch;color:var(--ink-2);margin:0 0 16px;}
p strong,li strong{color:var(--ink);font-weight:650;}
.fig{background:var(--surface);border:1px solid var(--hair);border-radius:14px;
  padding:26px clamp(16px,3vw,30px) 22px;margin:22px 0 10px;overflow-x:auto;}
.fig-title{font-size:15.5px;font-weight:650;color:var(--ink);margin:0 0 3px;}
.fig-note{font-size:13.5px;color:var(--muted);margin:2px 0 20px;max-width:76ch;}
.legend{display:flex;gap:20px;flex-wrap:wrap;margin:0 0 4px;font-size:13px;color:var(--ink-2);}
.legend .sw{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:7px;vertical-align:middle;}
.legend .dumb{display:inline-flex;align-items:center;gap:5px;}
.legend .ring{width:10px;height:10px;border-radius:50%;border:2px solid var(--muted);display:inline-block;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0;}
.tile{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:15px 16px 14px;}
.tile-name{font-size:14px;font-weight:650;color:var(--ink);}
.tile-sub{font-size:12px;color:var(--muted);margin-top:1px;}
.tile-vals{display:flex;gap:16px;margin-top:11px;align-items:baseline;}
.tv{font-family:ui-monospace,Menlo,monospace;font-size:27px;font-weight:600;font-variant-numeric:tabular-nums;}
.tv .pct{font-size:14px;font-weight:500;margin-left:1px;}
/* svg text classes */
.grid{stroke:var(--grid);stroke-width:1;}
.conn{stroke:var(--hair-strong);stroke-width:2.5;}
.conn2{stroke:var(--hair-strong);stroke-width:2;}
.tick{fill:var(--muted);font-size:11.5px;font-family:ui-monospace,Menlo,monospace;}
.rowlab{fill:var(--ink);font-size:14px;font-weight:500;}
.rowlab-sm{fill:var(--ink-2);font-size:13px;}
.rowlab-sm.shared{fill:var(--ink);font-weight:600;}
.eyebrow-svg{fill:var(--muted);font-size:11px;letter-spacing:.12em;font-family:ui-monospace,Menlo,monospace;}
.col-hd{font-size:14px;font-weight:700;letter-spacing:.01em;}
.val{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;font-weight:600;font-variant-numeric:tabular-nums;}
.barval{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;font-weight:600;}
.drop{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;font-weight:600;}
.muted-t{fill:var(--muted);} .warn{fill:var(--warn);}
.llama-t{fill:var(--llama);} .qwen-t{fill:var(--qwen);}
.llama-f{fill:var(--llama);} .qwen-f{fill:var(--qwen);}
.llama-fill{fill:var(--llama);} .qwen-fill{fill:var(--qwen);}
.dot{stroke:var(--surface);stroke-width:1.5;}
.dot-hollow{fill:var(--surface);stroke:var(--muted);stroke-width:2;}
ul{max-width:68ch;color:var(--ink-2);padding-left:20px;}
li{margin-bottom:8px;}
.tablewrap{margin-top:14px;font-size:13px;}
.tablewrap summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:13px;}
table{border-collapse:collapse;margin-top:12px;font-size:13px;width:auto;}
th,td{padding:5px 16px 5px 0;text-align:left;border-bottom:1px solid var(--hair);}
td.num,th.num{text-align:right;font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;}
.callout{background:var(--surface-2);border:1px solid var(--hair);border-left:3px solid var(--accent);
  border-radius:0 10px 10px 0;padding:16px 20px;margin:20px 0;}
.callout p{margin:0;color:var(--ink-2);}
.callout strong{color:var(--ink);}
.foot{margin-top:70px;padding-top:22px;border-top:1px solid var(--hair);font-size:13px;color:var(--muted);}
.foot .mono{font-size:12px;}
@media (max-width:560px){
  .tv{font-size:23px;}
}
"""

def legend_models(extra=""):
    return f'''<div class="legend">
  <span><span class="sw" style="background:var(--llama)"></span>Llama-3-8B-Instruct</span>
  <span><span class="sw" style="background:var(--qwen)"></span>Qwen3-32B</span>{extra}
</div>'''

HTMLDOC = f"""<title>Refusal Atlas — signal-dependent over-refusal</title>
<style>{CSS}</style>
<div class="wrap">
<p class="eyebrow">Refusal Atlas · in-progress read</p>
<h1>The over-refusal boundary changes depending on which internal signal you read it through</h1>
<p class="lede">We map where two instruction-tuned models over-refuse <em>benign</em> prompts — by topic and by single word — and test whether three different "refusal signals" draw the same boundary. They don't.</p>
<div class="meta">
  <span><b>Models</b> Llama-3-8B-Instruct · Qwen3-32B</span>
  <span><b>Substrate</b> 1,636 prompts</span>
  <span><b>Ground truth</b> independent Sonnet judge</span>
  <span><b>Status</b> both models scored · analysis in progress</span>
</div>

<section>
<p class="kicker">The setup</p>
<h2>One substrate, three signals, an independent judge</h2>
<p>Every prompt is scored by all three candidate refusal signals <em>and</em> by the model's actual sampled behavior. The three signals are the <strong>refusal direction</strong> (projection onto the diff-of-means refusal vector), the <strong>probe ensemble</strong> (a mass-mean linear probe combined across layers), and the <strong>output logit</strong> (the model's own probability of <em>starting</em> its reply with a refusal). Behavior is labeled by an independent Claude-Sonnet judge rather than an opener regex — because that regex shares its definition with the logit signal, and grading the logit with its own definition is circular.</p>
<p>The substrate mixes OR-Bench-Hard (10 sensitive-but-benign topic categories), XSTest, a toxic "should-refuse" anchor, and <strong>463 freshly generated single-word minimal pairs</strong> — leakage-free, disjoint from every fitting set, each changing exactly one word (e.g. <span class="mono">reduce&nbsp;→&nbsp;attack</span>) to isolate one word's causal pull.</p>
</section>

<section>
<p class="kicker">Finding 1 · the behavioral map</p>
<h2>Both refuse the toxic anchor and comply with benign controls — but Llama over-refuses the sensitive-topic middle ~2.2× as often</h2>
<p>The behavior orders exactly as it should: near-ceiling on the toxic anchor, near-floor on benign minimal pairs. The gap opens in the middle — the genuinely benign but sensitive-topic prompts. By the independent judge, Llama refuses <strong>78%</strong> of OR-Bench-Hard vs Qwen's <strong>35%</strong>.</p>
{stat_row()}
<p class="fig-note" style="margin-top:-4px">Judge-labeled refusal rate on each prompt source. Two numbers per tile: <span class="llama-t mono" style="font-weight:600">Llama</span> · <span class="qwen-t mono" style="font-weight:600">Qwen</span>.</p>

<div class="fig">
<div class="fig-title">Over-refusal is not just higher for Llama — the topic <em>profile</em> differs</div>
<div class="fig-note">Judge-labeled refusal rate on benign OR-Bench-Hard prompts, by topic. Llama sits near-ceiling across hate / self-harm / unethical / violence; Qwen elevates mainly on violence. The line is the model gap.</div>
{legend_models()}
{fig_topics()}
{data_table()}
</div>
</section>

<section>
<p class="kicker">Finding 2 · the headline</p>
<h2>The refusal direction is partly a topic detector; the output logit is not</h2>
<p>Across the whole substrate all three signals separate refuse from comply well (AUC ≈ 0.97–0.99 vs the judge). But the boundary that matters for over-refusal lives <em>inside</em> a topic — same subject, benign vs not. Restricted to within-topic, the <strong>refusal-direction</strong> AUC drops (Qwen 0.969→<strong>0.874</strong>; Llama 0.977→0.946) while the <strong>output logit</strong> holds flat (Qwen 0.977→0.980; Llama 0.990→0.993). Much of the direction's apparent skill was reading the topic, not the refusal.</p>
<div class="fig">
<div class="fig-title">Full-substrate vs within-topic AUC, per signal</div>
<div class="fig-note">Hollow dot = full-substrate AUC; filled dot = within-topic AUC; the number at left is the drop (percentage points). A big drop means the signal was leaning on topic identity. Axis zoomed to 0.85–1.00.</div>
<div class="legend">
  <span class="dumb"><span class="ring"></span>full substrate</span>
  <span class="dumb"><span class="sw" style="background:var(--ink-2)"></span>within topic</span>
</div>
{fig_fidelity()}
</div>
<div class="callout"><p><strong>Why it matters:</strong> if you localize or steer refusal through the direction vector, part of what you're moving is "this is a violence-adjacent topic," not "this specific request should be refused." Reading the same boundary through the logit gives a materially different answer inside a topic.</p></div>
</section>

<section>
<p class="kicker">Finding 3 · the correction</p>
<h2>Circularity was real — and it inflated the logit most</h2>
<p>Our regex refusal detector and the logit signal share one definition of "refusal," so grading the logit against the regex flatters it. Swapping in the independent judge shrinks every signal's AUC, and shrinks the <strong>logit's</strong> most — exactly the signal the circularity predicts. The judge also shows the regex <em>under-counts</em> real refusals (it misses ~30–50 of ~1,230 unique replies per model), so raw behavioral rates are slight underestimates, not overestimates.</p>
<div class="fig">
<div class="fig-title">AUC inflation from the circular regex label (regex AUC − judge AUC)</div>
<div class="fig-note">How much each signal's fidelity was overstated by grading against the opener regex instead of the independent judge. The logit — which shares the regex's definition — is inflated most, especially for Qwen (+2.0pp).</div>
{legend_models()}
{fig_circularity()}
</div>
</section>

<section>
<p class="kicker">Finding 4 · word-level</p>
<h2>Single words move the boundary, and a shared core generalizes across both models</h2>
<p>On the leakage-free single-word pairs, changing one word flips an otherwise-benign task into a refusal. Five words — <strong>attack, contraband, exploit, lethal, weaponize</strong> (✦) — do it for <em>both</em> models; others are model-specific (Llama: <span class="mono">interrogation</span>; Qwen: <span class="mono">manipulation</span>, <span class="mono">dissect</span>).</p>
<div class="fig">
<div class="fig-title">Top single-word triggers — change in judge refusal rate, one word swapped</div>
<div class="fig-note">Δ refusal (percentage points) attributable to a single substituted word, averaged over its minimal pairs. ✦ marks words that trigger both models.</div>
{legend_models('<span><span class="sw" style="background:var(--muted)">&#10022;</span>shared trigger</span>')}
{fig_triggers()}
</div>
</section>

<section>
<p class="kicker">Method safeguards</p>
<h2>What keeps these clean</h2>
<ul>
<li><strong>Independent ground truth.</strong> Behavior is judged by Claude-Sonnet on a rubric with no shared definition with any signal — de-circularizing the whole comparison.</li>
<li><strong>Leakage-free pairs.</strong> The 463 single-word pairs are generated from prompts disjoint from every probe/vector fitting set; both models are seeing them for the first time.</li>
<li><strong>Causal layer validation.</strong> The refusal direction is taken at the causally-controlling layer (Llama L17, validated by ablation; Qwen L58, validated by addition — driving a benign prompt to 93% refusal), not merely the most predictive one.</li>
<li><strong>Adversarial code review before spend.</strong> Method/code reviews caught real bugs pre-run (a padding bug that would have corrupted the behavioral ground truth; a wrong-layer fallback).</li>
</ul>
</section>

<section>
<p class="kicker">Next</p>
<h2>Where it goes from here</h2>
<ul>
<li><strong>Topic clustering map (P5)</strong> — unsupervised structure over the substrate, beyond the fixed OR-Bench 10 categories, with bootstrap-Jaccard stability.</li>
<li><strong>Per-trigger-word deep dive (P6)</strong> — the full 463-pair word-effect atlas with confidence intervals, not just the top words.</li>
<li><strong>Statistics hardening</strong> — bootstrap CIs and a noise null on every AUC and rate reported here.</li>
</ul>
</section>

<div class="foot">
<p>Refusal Atlas · figures generated from the independent-judge scoring pass on both models.<br>
<span class="mono">substrate: 1,636 prompts · behavioral n=8 samples/prompt · ground truth: claude-sonnet judge · all metrics judge-labeled unless noted</span></p>
</div>
</div>
"""

out = os.path.join(HERE, "figures.html")
open(out, "w").write(HTMLDOC)
print("[wrote]", out, f"({len(HTMLDOC)} bytes)")
