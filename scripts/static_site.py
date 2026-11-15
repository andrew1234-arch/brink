"""Render the Brink product page as one static HTML file (a Hugging Face Space with `sdk: static`).

The page is for someone who wants to use Brink: what it does, how to put it in a fine-tuning workflow (the
`brink` command, the training callback, the CuratorKIT gate), and an interactive demo driven by a recorded run:
move the alarm threshold or the gate's drop share and see what Brink would have done. Every number on the page
is read from the result files next to it. The experiment reports sit in a collapsed evidence section.

A static page runs no model. The demo replays logged signals; scoring your own data and running Brink on your
own model happen with the `brink` command or the Gradio app (`make app`).

Only a small subset of Markdown is rendered, the subset the reports use: headings, paragraphs, bullet lists,
pipe tables, **bold**, `code` and [links](url). Everything is HTML-escaped first.
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path

REPO_URL = "https://github.com/andrew1234-arch/brink"
SITE_URL = "https://cmgdevl-brink.static.hf.space"
SPACE_URL = "https://huggingface.co/spaces/cmgdevl/brink"
ADAPTER_REPO = "cmgdevl/brink-artifacts"
ADAPTER_DIR = "adapters/_work/pipeline/adapters/brink"
RESULTS_REPO = "cmgdevl/brink-results"
TITLE = "Brink · Keep a model safe while you fine-tune it"
DESCRIPTION = (
    "Open-source safety guard for LLM fine-tuning. Brink removes risky samples from your training data, watches "
    "the model's refusals while it trains, and stops the run before its safety breaks."
)
BRAND = "#7459CC"
MARK_SVG = (
    '<svg viewBox="0 0 64 64" width="28" height="28" aria-hidden="true"><defs><clipPath id="bm">'
    '<rect width="64" height="64" rx="15"/></clipPath></defs><g clip-path="url(#bm)">'
    '<rect width="64" height="64" fill="#7459CC"/><rect x="41" y="51" width="23" height="13" fill="#4A3699"/>'
    '<path d="M0 37h41v27H0z" fill="#2B2063"/><circle cx="34" cy="30" r="7" fill="#FFFFFF"/></g></svg>'
)
ORDER = ["live", "pipeline", "go_nogo", "gate", "repair", "multilingual", "circuit"]
TITLES = {
    "live": "Brink end to end (recorded live run)",
    "pipeline": "Pipeline: a Hindi assistant, before and after",
    "go_nogo": "Go / no-go: is the drift detectable?",
    "gate": "Gate: finding the data that erodes safety",
    "repair": "Repair: rollback and the erosion guard vs baselines",
    "multilingual": "Multilingual: Hindi and Tagalog",
    "circuit": "CircuitKIT: the refusal circuit vs the monitored layer",
}


def _inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<em>\1</em>", t)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', t)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def markdown_to_html(md: str, shift: int = 0) -> str:
    """Render the reports' Markdown subset. `shift` lowers every heading by that many levels."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
        elif m := re.match(r"(#{1,6})\s+(.*)", line):
            level = min(len(m.group(1)) + shift, 6)
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            has_head = len(rows) > 1 and all(re.fullmatch(r":?-+:?", c) for c in rows[1])
            body = rows[2:] if has_head else rows
            table = "<div class='scroll'><table>"
            if has_head:
                table += "<thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in rows[0]) + "</tr></thead>"
            table += "<tbody>"
            for row in body:
                table += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
            out.append(table + "</tbody></table></div>")
        elif re.match(r"\s*[-*]\s+", line):
            items = []
            while i < len(lines) and (re.match(r"\s*[-*]\s+", lines[i]) or (items and lines[i].startswith("  "))):
                if re.match(r"\s*[-*]\s+", lines[i]):
                    items.append(re.sub(r"^\s*[-*]\s+", "", lines[i].rstrip()))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"#{1,6}\s|\s*\||\s*[-*]\s+", lines[i]):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{_inline(' '.join(para))}</p>")
    return "\n".join(out)


INSTALL = 'pip install "brink-safety[curatorkit] @ git+https://github.com/andrew1234-arch/brink"'

GATE_CLI = """brink gate my_finetune.jsonl --model CohereLabs/tiny-aya-global --drop 0.05 --out gated/

# gated/scored.csv     every sample with its risk score, riskiest first
# gated/cleaned.jsonl  your data without the riskiest 5 % and exact duplicates"""

CALLBACK = """from brink.monitor.callback import BrinkCallback
from brink.monitor.core import Monitor
from brink.probe import Prober, calibrate
from brink.repair import ErosionGuard, RepairController

fp, _ = calibrate(model, tok, harmful_cal, harmless_cal, harmful_probe, harmless_probe)

monitor = Monitor(Prober(fp, tok, harmful_probe, harmless_probe), "runs/my-finetune",
                  probe_every=4, alarm_signal="S3", alarm_threshold=-3.0)
controller = RepairController(monitor, "halt")

trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                     callbacks=[BrinkCallback(controller=controller)])
trainer.train()"""

CURATOR = """from curatorkit.pipeline import Pipeline
from brink.gate import GateScorer
from brink.gate.curatorkit_gate import BrinkSafetyGate

gate = BrinkSafetyGate(GateScorer(model, tok, fp, harmful_probe, harmless_probe), drop_frac=0.05)
Pipeline([reader, ExactDeduplicator(), gate, exporter], output_dir="curated").run()"""

RUN_CLI = """brink run --model Qwen/Qwen2.5-0.5B-Instruct --out runs/demo

# fingerprints the model, gates a contaminated set, fine-tunes it unprotected and with Brink,
# verifies both on held-out prompts and writes runs/demo/REPORT.md"""

ADAPTER = f"""from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global")
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
model = PeftModel.from_pretrained(base, "{ADAPTER_REPO}", subfolder="{ADAPTER_DIR}")"""

APP = """git clone https://github.com/andrew1234-arch/brink && cd brink
pip install -e ".[dashboard,curatorkit]"
make app        # http://localhost:7860: Run Brink, Gate your data, Experiments"""

CSS = """
:root { --bg: #f8f7fc; --card: #ffffff; --fg: #1a1726; --muted: #5d5a6e; --line: #e2dff0; --accent: #5b43b3;
  --brand: #7459cc; --brand-fg: #ffffff; --tint: #eeeafd; --bad: #c2410c; --good: #0f766e; --code-bg: #17142a;
  --code-fg: #ebe8f7; --soft: #f1eefb; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #121118; --card: #1b1924; --fg: #ece9f6; --muted: #a3a0b5; --line: #2d2a3b; --accent: #bcaef7;
    --brand: #bcaef7; --brand-fg: #121118; --tint: #241f3a; --bad: #fb923c; --good: #5eead4; --code-bg: #0c0b12;
    --code-fg: #e2def2; --soft: #221f2e; }
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 64px; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.6 system-ui, -apple-system, "Segoe UI",
  sans-serif; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); }
a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible {
  outline: 2px solid var(--brand); outline-offset: 2px; border-radius: 4px; }
.skip { position: absolute; left: -999px; top: 8px; background: var(--card); padding: 8px 12px; z-index: 10; }
.skip:focus { left: 16px; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 0 16px; }
header.top { position: sticky; top: 0; background: color-mix(in srgb, var(--bg) 92%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); z-index: 5; }
header.top .wrap { display: flex; align-items: center; gap: 20px; height: 56px; }
.brand { display: flex; align-items: center; gap: 9px; font-weight: 750; letter-spacing: -0.01em;
  text-decoration: none; color: var(--fg); font-size: 1.12rem; }
.brand svg { display: block; border-radius: 7px; }
header.top nav { display: flex; gap: 18px; margin-left: auto; overflow-x: auto; white-space: nowrap; }
header.top nav a { color: var(--muted); text-decoration: none; font-size: 0.93rem; }
header.top nav a:hover { color: var(--fg); }
.hero { padding-top: 64px; padding-bottom: 40px; }
.eyebrow { display: inline-block; font-size: 0.82rem; font-weight: 650; color: var(--accent); background: var(--tint);
  border-radius: 999px; padding: 3px 12px; margin: 0 0 16px; }
.hero h1 { font-size: clamp(2.1rem, 5.4vw, 3.4rem); line-height: 1.06; letter-spacing: -0.03em; margin: 0 0 18px;
  max-width: 22ch; }
.hero h1 em { font-style: normal; color: var(--brand); }
.hero p.lead { font-size: 1.15rem; color: var(--muted); max-width: 62ch; margin: 0 0 26px; }
.cta { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }
.btn { display: inline-block; padding: 10px 18px; border-radius: 9px; text-decoration: none; font-weight: 650;
  border: 1px solid var(--line); color: var(--fg); background: var(--card); }
.btn:hover { border-color: var(--brand); }
.btn.primary { background: var(--brand); color: var(--brand-fg); border-color: var(--brand); }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(176px, 1fr)); gap: 12px; margin: 28px 0 8px; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }
.stat b { display: block; font-size: 1.4rem; letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
  color: var(--fg); white-space: nowrap; }
.stat span { color: var(--muted); font-size: 0.88rem; line-height: 1.45; display: block; margin-top: 4px; }
.note { color: var(--muted); font-size: 0.88rem; margin: 8px 0 0; }
section.block { padding: 48px 0; border-top: 1px solid var(--line); }
section.block > .wrap > h2 { font-size: 1.65rem; letter-spacing: -0.02em; margin: 0 0 6px; }
section.block > .wrap > p.sub { color: var(--muted); margin: 0 0 22px; max-width: 70ch; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.step { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px;
  border-top: 3px solid var(--brand); }
.step .when { color: var(--accent); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; }
.step h3 { margin: 4px 0 6px; font-size: 1.12rem; }
.step p { margin: 0; color: var(--muted); font-size: 0.95rem; }
.use { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 20px; margin: 0 0 14px; }
.use h3 { margin: 0 0 4px; font-size: 1.1rem; }
.use p { margin: 0 0 12px; color: var(--muted); font-size: 0.95rem; max-width: 75ch; }
.code { position: relative; }
pre { margin: 0; background: var(--code-bg); color: var(--code-fg); border-radius: 9px; padding: 14px 76px 14px 16px;
  overflow-x: auto; font: 0.86rem/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.copy { position: absolute; top: 8px; right: 8px; font: inherit; font-size: 0.78rem; padding: 3px 9px;
  border-radius: 6px; border: 1px solid #3b3557; background: #26213d; color: #ebe8f7; cursor: pointer; }
.copy:hover { border-color: #bcaef7; }
code { font: 0.9em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--soft);
  border-radius: 4px; padding: 1px 5px; }
pre code { background: none; padding: 0; }
.demo { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 20px; margin: 0 0 14px; }
.demo h3 { margin: 0 0 4px; font-size: 1.1rem; }
.demo p.q { margin: 0 0 14px; color: var(--muted); font-size: 0.95rem; max-width: 75ch; }
.controls { display: flex; flex-wrap: wrap; gap: 18px 28px; align-items: center; margin: 0 0 10px; }
.controls label { font-size: 0.9rem; color: var(--muted); display: flex; align-items: center; gap: 10px; }
.controls output { color: var(--fg); font-weight: 650; font-variant-numeric: tabular-nums; min-width: 3.5ch; }
input[type=range] { accent-color: var(--brand); width: 200px; max-width: 46vw; }
select { font: inherit; padding: 3px 6px; border-radius: 6px; border: 1px solid var(--line); background: var(--card);
  color: var(--fg); }
svg.chart { width: 100%; height: auto; display: block; }
.verdict { margin: 10px 0 0; padding: 10px 14px; border-left: 3px solid var(--brand); background: var(--tint);
  border-radius: 0 8px 8px 0; font-size: 0.97rem; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.85rem; color: var(--muted); margin: 6px 0 0; }
.legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px;
  vertical-align: -1px; }
ul.plain { margin: 0; padding-left: 20px; color: var(--muted); max-width: 80ch; }
ul.plain li { margin: 6px 0; }
details.ev { background: var(--card); border: 1px solid var(--line); border-radius: 12px; margin: 0 0 10px; }
details.ev > summary { cursor: pointer; padding: 14px 18px; font-weight: 650; display: flex; gap: 12px;
  align-items: center; flex-wrap: wrap; }
details.ev > div { padding: 0 18px 18px; }
.chip { font-size: 0.76rem; font-weight: 650; padding: 2px 9px; border-radius: 999px; background: var(--soft);
  color: var(--muted); }
.chip.ok { color: var(--good); } .chip.no { color: var(--bad); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 10px 0; font-size: 0.9rem; }
th, td { border: 1px solid var(--line); padding: 5px 9px; text-align: left; vertical-align: top; }
img { max-width: 100%; height: auto; background: #fff; border-radius: 6px; margin: 10px 0; }
.files { color: var(--muted); font-size: 0.88rem; }
h4, h5, h6 { margin: 16px 0 6px; font-size: 1rem; }
.links { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.links a { display: block; background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px;
  text-decoration: none; color: var(--fg); }
.links a:hover { border-color: var(--brand); }
.links b { display: block; margin-bottom: 2px; }
.links span { color: var(--muted); font-size: 0.9rem; }
footer { border-top: 1px solid var(--line); padding: 26px 0 50px; color: var(--muted); font-size: 0.9rem; }
footer .wrap { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; }
.banner { background: var(--soft); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px;
  margin: 18px 0 0; }
@media (max-width: 640px) { header.top nav a.minor { display: none; } .hero { padding-top: 40px; } }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
"""

SCRIPT = """
(function () {
  var D = window.BRINK_DEMO;
  var NS = "http://www.w3.org/2000/svg";
  function el(name, attrs, text) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function firstAlarm(steps, z, thr, persist) {
    var run = 0;
    for (var i = 0; i < z.length; i++) {
      run = z[i] <= thr ? run + 1 : 0;
      if (run >= persist) return steps[i];
    }
    return null;
  }
  document.querySelectorAll(".copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var text = b.parentNode.querySelector("pre").textContent;
      navigator.clipboard.writeText(text).then(function () {
        b.textContent = "Copied";
        setTimeout(function () { b.textContent = "Copy"; }, 1400);
      });
    });
  });
  if (!D) return;

  function drawAlarm() {
    var svg = document.getElementById("alarm-chart");
    var thr = parseFloat(document.getElementById("thr").value);
    var persist = parseInt(document.getElementById("persist").value, 10);
    document.getElementById("thr-out").textContent = thr.toFixed(1);
    var W = 960, H = 340, L = 52, R = 110, T = 16, B = 40;
    var all = D.unprotected.z.concat(D.brink.z);
    var lo = Math.min.apply(null, all.concat([thr])) - 2, hi = Math.max.apply(null, all) + 2;
    var maxStep = Math.max.apply(null, D.unprotected.steps.concat(D.brink.steps));
    function x(s) { return L + (W - L - R) * s / maxStep; }
    function y(v) { return T + (H - T - B) * (hi - v) / (hi - lo); }
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var line = css("--line"), muted = css("--muted"), brand = css("--brand");
    for (var g = Math.ceil(lo / 10) * 10; g <= hi; g += 10) {
      svg.appendChild(el("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), stroke: line }));
      svg.appendChild(el("text", { x: L - 8, y: y(g) + 4, "text-anchor": "end", fill: muted, "font-size": 12 }, g));
    }
    D.unprotected.steps.forEach(function (s) {
      if (s % 8 === 0) {
        svg.appendChild(el("text", { x: x(s), y: H - B + 18, "text-anchor": "middle", fill: muted,
          "font-size": 12 }, s));
      }
    });
    svg.appendChild(el("text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", fill: muted,
      "font-size": 12 }, "training step"));
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: y(thr), y2: y(thr), stroke: muted,
      "stroke-dasharray": "6 5" }));
    svg.appendChild(el("text", { x: W - R + 6, y: y(thr) + 4, fill: muted, "font-size": 12 }, "alarm line"));
    var out = [];
    [["unprotected", css("--bad"), "Unprotected"], ["brink", brand, "With Brink"]].forEach(function (c) {
      var s = D[c[0]], pts = s.steps.map(function (st, i) { return x(st) + "," + y(s.z[i]); }).join(" ");
      svg.appendChild(el("polyline", { points: pts, fill: "none", stroke: c[1], "stroke-width": 2.5 }));
      s.steps.forEach(function (st, i) {
        svg.appendChild(el("circle", { cx: x(st), cy: y(s.z[i]), r: 3, fill: c[1] }));
      });
      var a = firstAlarm(s.steps, s.z, thr, persist);
      if (a !== null) {
        svg.appendChild(el("line", { x1: x(a), x2: x(a), y1: T, y2: H - B, stroke: c[1], "stroke-width": 1.5 }));
        svg.appendChild(el("text", { x: x(a) + 5, y: T + 12 + (c[0] === "brink" ? 16 : 0), fill: c[1],
          "font-size": 12 }, c[2] + ": alarm at step " + a));
      }
      out.push([c[2], a]);
    });
    var u = out[0][1], b = out[1][1], msg;
    if (u === null) {
      msg = "With this rule the alarm never fires on the unprotected run, and the model ends up answering " +
        D.asr_unprotected + " of harmful prompts. The line is too low.";
    } else {
      msg = "The alarm fires at step " + u + " of " + D.max_step + " on the unprotected run, which ended at " +
        D.asr_unprotected + " attack success. ";
      msg += b === null
        ? "It stays quiet on the Brink run, which ended at " + D.asr_brink + "."
        : "It also fires on the Brink run at step " + b + ", where the model was still safe (" + D.asr_brink +
          "): a false alarm. The line is too sensitive.";
    }
    document.getElementById("alarm-verdict").textContent = msg;
  }

  function drawGate() {
    var svg = document.getElementById("gate-chart");
    var frac = parseFloat(document.getElementById("drop").value);
    var n = D.gate.length, k = Math.round(frac * n);
    document.getElementById("drop-out").textContent = Math.round(frac * 100) + " %";
    var W = 960, H = 220, L = 52, R = 16, T = 12, B = 30;
    var scores = D.gate.map(function (g) { return g[0]; });
    var hi = Math.max.apply(null, scores), lo = Math.min.apply(null, scores.concat([0]));
    function y(v) { return T + (H - T - B) * (hi - v) / (hi - lo); }
    var bw = (W - L - R) / n;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var bad = css("--bad"), good = css("--good"), muted = css("--muted"), line = css("--line");
    if (k > 0) {
      svg.appendChild(el("rect", { x: L, y: T, width: bw * k, height: H - T - B, fill: line, opacity: 0.55 }));
      svg.appendChild(el("text", { x: L + 6, y: H - B - 8, fill: muted, "font-size": 12 }, "dropped by the gate"));
    }
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: muted }));
    svg.appendChild(el("text", { x: L - 8, y: y(0) + 4, "text-anchor": "end", fill: muted, "font-size": 12 }, "0"));
    svg.appendChild(el("text", { x: L - 8, y: y(hi) + 10, "text-anchor": "end", fill: muted, "font-size": 12 },
      hi.toFixed(2)));
    var caught = 0, benign = 0, total = 0;
    D.gate.forEach(function (g, i) {
      var top = Math.min(y(g[0]), y(0)), h = Math.max(Math.abs(y(g[0]) - y(0)), 1);
      svg.appendChild(el("rect", { x: L + bw * i + 0.5, y: top, width: Math.max(bw - 1, 1), height: h,
        fill: g[1] ? bad : good }));
      total += g[1];
      if (i < k) { if (g[1]) caught++; else benign++; }
    });
    svg.appendChild(el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", fill: muted,
      "font-size": 12 }, "training samples, riskiest first"));
    document.getElementById("gate-verdict").textContent = k === 0
      ? "Nothing dropped: all " + total + " harmful samples go into training."
      : "Dropping the riskiest " + k + " of " + n + " samples removes " + caught + " of the " + total +
        " harmful ones and costs " + benign + " benign sample" + (benign === 1 ? "" : "s") + ". Dropping " + k +
        " at random would remove about " + Math.round(k * total / n) + ".";
  }

  ["thr", "persist"].forEach(function (id) { document.getElementById(id).addEventListener("input", drawAlarm); });
  document.getElementById("drop").addEventListener("input", drawGate);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      drawAlarm(); drawGate();
    });
  }
  drawAlarm(); drawGate();
})();
"""


def _ordered(stages: list[str]) -> list[str]:
    return [s for s in ORDER if s in stages] + sorted(s for s in stages if s not in ORDER)


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * float(v):.0f} %"


def demo_data(results: Path) -> dict | None:
    """The recorded end-to-end run, reduced to what the interactive demo draws. None if it isn't there."""
    live = results / "live"
    summary = _json(live / "summary.json")
    series = {}
    for cond in ("unprotected", "brink"):
        rows = [r for r in _rows(live / cond / "probe.csv") if r.get("z_S3") not in ("", None)]
        if not rows:
            return None
        series[cond] = {"steps": [int(float(r["step"])) for r in rows], "z": [float(r["z_S3"]) for r in rows]}
    gate = [[float(r["score"]), int(r["harmful"])] for r in _rows(live / "gate.csv")]
    if not summary or not gate:
        return None
    cfg = summary.get("config") or {}
    return {
        **series,
        "gate": gate,
        "threshold": float(cfg.get("alarm_threshold", -3.0)),
        "persistence": int(cfg.get("alarm_persistence", 2)),
        "drop": float(cfg.get("gate_drop_frac", 0.25)),
        "max_step": max(series["unprotected"]["steps"]),
        "model": summary.get("model", ""),
        "asr_before": _pct((summary.get("before") or {}).get("asr")),
        "asr_unprotected": _pct((summary.get("unprotected") or {}).get("asr")),
        "asr_brink": _pct((summary.get("brink") or {}).get("asr")),
    }


def _stats(results: Path) -> str:
    """Headline numbers, each read from a summary.json. Empty if the recorded run isn't there."""
    live, pipe = _json(results / "live" / "summary.json"), _json(results / "pipeline" / "summary.json")
    tiles = []
    if live.get("unprotected") and live.get("brink"):
        u, b, g = live["unprotected"], live["brink"], live.get("gate") or {}
        tiles.append((f"{_pct(u.get('asr'))} → {_pct(b.get('asr'))}", "harmful requests answered after the same "
                      "fine-tune, without and with Brink"))  # fmt: skip
        if g:
            tiles.append((f"{g['harmful_caught']} of {g['harmful_total']}", "harmful training samples removed by "
                          "the gate before training"))  # fmt: skip
        tiles.append((_pct(b.get("over_refusal")), "harmless requests refused by the protected model"))
        if u.get("heldout_loss") is not None and b.get("heldout_loss") is not None:
            tiles.append((f"{u['heldout_loss']:.2f} vs {b['heldout_loss']:.2f}", "task loss without and with Brink: "
                          "the model still learns the task"))  # fmt: skip
    conds = pipe.get("conditions") or {}
    if conds.get("naive") and conds.get("brink"):
        n, b = conds["naive"]["final"], conds["brink"]["final"]
        tiles.append((f"{_pct(n.get('asr'))} → {_pct(b.get('asr'))}", "the same on a Hindi assistant trained through "
                      "CuratorKIT and AlignTune"))  # fmt: skip
    if not tiles:
        return ""
    model = html.escape(live.get("model") or "Tiny Aya")
    cells = "".join(f'<div class="stat"><b>{html.escape(v)}</b><span>{html.escape(t)}</span></div>' for v, t in tiles)
    return (
        f'<div class="stats">{cells}</div><p class="note">Measured on <code>{model}</code> in recorded runs. '
        'Full reports, including what did not work, are under <a href="#evidence">Evidence</a>.</p>'
    )


def _code(text: str) -> str:
    return (
        '<div class="code"><button class="copy" type="button">Copy</button>'
        f"<pre><code>{html.escape(text)}</code></pre></div>"
    )


def _verdict_chip(report: str) -> str:
    m = re.search(r"^#\s+.*?\*\*(.+?)\*\*", report, flags=re.MULTILINE)
    if not m:
        return ""
    word = m.group(1)
    cls = "no" if re.search(r"NO-GO|NOT", word) else "ok"
    return f'<span class="chip {cls}">{html.escape(word)}</span>'


def _evidence(results: Path, stages: list[str]) -> str:
    out = []
    for s in stages:
        folder = results / s
        report = (folder / "REPORT.md").read_text(encoding="utf-8")
        plots = "".join(
            f'<img src="results/{s}/{p.name}" alt="{html.escape(s)} plot: {html.escape(p.stem)}" loading="lazy">'
            for p in sorted(folder.glob("*.png"))
        )
        files = ", ".join(
            f'<a href="results/{s}/{f.name}">{html.escape(f.name)}</a>'
            for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix in {".json", ".csv", ".md"}
        )
        out.append(
            f'<details class="ev" id="{html.escape(s)}"><summary>{html.escape(TITLES.get(s, s))}'
            f"{_verdict_chip(report)}</summary><div>{markdown_to_html(report, shift=3)}{plots}"
            f"<p class='files'>Files: {files}</p></div></details>"
        )
    roi = results / "ROI.md"
    if roi.is_file():
        out.append(
            '<details class="ev" id="roi"><summary>Cost and ROI</summary><div>'
            f"{markdown_to_html(roi.read_text(encoding='utf-8'), shift=3)}</div></details>"
        )
    return "".join(out)


def _demo(d: dict | None) -> str:
    if d is None:
        return ""
    model = html.escape(d["model"])
    return f"""
<section class="block" id="demo"><div class="wrap">
<h2>Try it on a recorded run</h2>
<p class="sub">This is a real fine-tune of <code>{model}</code> on data with harmful samples mixed in, logged step
by step. Move the controls to see what Brink would have done with your settings.</p>

<div class="demo">
<h3>Where would the alarm fire?</h3>
<p class="q">Brink reads the model's preference to refuse every few steps, as a z-score against the model before
fine-tuning. The alarm fires when it stays under the line.</p>
<div class="controls">
<label>Alarm line (z) <input id="thr" type="range" min="-12" max="-1" step="0.5" value="{d["threshold"]}">
<output id="thr-out"></output></label>
<label>Probes in a row <select id="persist">{
        "".join(f'<option value="{k}"{" selected" if k == d["persistence"] else ""}>{k}</option>' for k in (1, 2, 3))
    }</select></label>
</div>
<svg class="chart" id="alarm-chart" viewBox="0 0 960 340" role="img"
  aria-label="Refusal signal over training steps for the unprotected and the Brink run"></svg>
<p class="verdict" id="alarm-verdict"></p>
</div>

<div class="demo">
<h3>How much should the gate drop?</h3>
<p class="q">Before training, Brink scores every sample by how much one training step on it would erode refusal.
Each bar is one training sample of this run.</p>
<div class="controls">
<label>Drop the riskiest <input id="drop" type="range" min="0" max="0.5" step="0.01" value="{d["drop"]}">
<output id="drop-out"></output></label>
</div>
<svg class="chart" id="gate-chart" viewBox="0 0 960 220" role="img"
  aria-label="Gate risk score of every training sample, riskiest first"></svg>
<div class="legend"><span><i style="background:var(--bad)"></i>harmful sample</span>
<span><i style="background:var(--good)"></i>benign sample</span></div>
<p class="verdict" id="gate-verdict"></p>
</div>
<p class="note">The demo replays logged signals in your browser. To score your own data or run your own model, use
the commands under <a href="#use">Use it</a>.</p>
</div></section>"""


def manifest() -> str:
    """The web app manifest: name, colours and icons for installs and Android home screens."""
    icons = [
        {"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "icon-512.png", "sizes": "512x512", "type": "image/png"},
        {"src": "icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
        {"src": "favicon.svg", "sizes": "any", "type": "image/svg+xml"},
    ]
    return json.dumps(
        {
            "name": "Brink",
            "short_name": "Brink",
            "description": DESCRIPTION,
            "start_url": "./",
            "scope": "./",
            "display": "standalone",
            "background_color": "#f8f7fc",
            "theme_color": BRAND,
            "icons": icons,
        },
        indent=2,
    )


def robots() -> str:
    return f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}/sitemap.xml\n"


def sitemap(lastmod: str | None = None) -> str:
    import datetime

    day = lastmod or datetime.date.today().isoformat()
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{SITE_URL}/</loc><lastmod>{day}</lastmod><changefreq>weekly</changefreq>"
        "<priority>1.0</priority></url>\n</urlset>\n"
    )


def structured_data() -> str:
    """JSON-LD for search engines: the software, its source code and the page."""
    graph = [
        {
            "@type": "SoftwareApplication",
            "@id": f"{SITE_URL}/#software",
            "name": "Brink",
            "description": DESCRIPTION,
            "applicationCategory": "DeveloperApplication",
            "applicationSubCategory": "AI safety",
            "operatingSystem": "Linux, macOS, Windows",
            "softwareVersion": "0.1.0",
            "license": "https://opensource.org/licenses/MIT",
            "url": f"{SITE_URL}/",
            "image": f"{SITE_URL}/og-image.png",
            "downloadUrl": REPO_URL,
            "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"},
            "keywords": "LLM fine-tuning, AI safety, alignment, refusal, LoRA, data curation, red teaming",
        },
        {
            "@type": "SoftwareSourceCode",
            "@id": f"{SITE_URL}/#code",
            "name": "Brink",
            "codeRepository": REPO_URL,
            "programmingLanguage": "Python",
            "runtimePlatform": "Python 3.10+",
            "license": "https://opensource.org/licenses/MIT",
            "targetProduct": {"@id": f"{SITE_URL}/#software"},
        },
        {
            "@type": "WebPage",
            "@id": f"{SITE_URL}/#page",
            "url": f"{SITE_URL}/",
            "name": TITLE,
            "description": DESCRIPTION,
            "inLanguage": "en",
            "primaryImageOfPage": f"{SITE_URL}/og-image.png",
            "about": {"@id": f"{SITE_URL}/#software"},
        },
    ]
    return json.dumps({"@context": "https://schema.org", "@graph": graph}, indent=1).replace("</", "<\\/")


def _head() -> str:
    e = html.escape
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(TITLE)}</title>
<meta name="description" content="{e(DESCRIPTION)}">
<meta name="robots" content="index, follow, max-image-preview:large">
<link rel="canonical" href="{SITE_URL}/">
<meta name="author" content="Cmg">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#f8f7fc" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#121118" media="(prefers-color-scheme: dark)">
<link rel="icon" href="favicon.ico" sizes="48x48">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="icon" href="favicon-32.png" type="image/png" sizes="32x32">
<link rel="icon" href="favicon-16.png" type="image/png" sizes="16x16">
<link rel="apple-touch-icon" href="apple-touch-icon.png">
<link rel="manifest" href="site.webmanifest">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Brink">
<meta property="og:title" content="{e(TITLE)}">
<meta property="og:description" content="{e(DESCRIPTION)}">
<meta property="og:url" content="{SITE_URL}/">
<meta property="og:locale" content="en_US">
<meta property="og:image" content="{SITE_URL}/og-image.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Brink: keep a model safe while you fine-tune it">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{e(TITLE)}">
<meta name="twitter:description" content="{e(DESCRIPTION)}">
<meta name="twitter:image" content="{SITE_URL}/og-image.png">
<meta name="twitter:image:alt" content="Brink: keep a model safe while you fine-tune it">
<script type="application/ld+json">{structured_data()}</script>"""


def render(results: Path, stages: list[str], simulated: bool = False) -> str:
    """The whole page. `results` is the folder published next to index.html as `results/`."""
    stages = _ordered(stages)
    demo = None if simulated else demo_data(results)
    banner = (
        '<div class="banner"><strong>SIMULATED runs.</strong> No real experiment has finished yet; the evidence '
        "section shows generated examples of the report format, not measurements.</div>"
        if simulated
        else ""
    )
    payload = json.dumps(demo).replace("</", "<\\/") if demo else ""
    data_script = f"<script>window.BRINK_DEMO = {payload};</script>" if demo else ""
    return f"""<!doctype html>
<html lang="en">
<head>
{_head()}
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top"><div class="wrap">
<a class="brand" href="#top" aria-label="Brink, home">{MARK_SVG}<span>Brink</span></a>
<nav><a class="minor" href="#how">How it works</a><a href="#use">Use it</a>{'<a href="#demo">Demo</a>' if demo else ""}
<a class="minor" href="#limits">Limits</a><a class="minor" href="#evidence">Evidence</a>
<a href="#resources">Resources</a></nav>
</div></header>

<main id="main">
<div class="wrap hero" id="top">
<p class="eyebrow">Open-source safety guard for LLM fine-tuning</p>
<h1>Keep a model safe <em>while</em> you fine-tune it.</h1>
<p class="lead">Fine-tuning can quietly teach a safe model to answer harmful requests. Brink cleans the risky
samples out of your training data, watches the model's refusals while it trains, and stops the run when they
start to slip.</p>
<div class="cta"><a class="btn primary" href="#use">Get started</a>
{'<a class="btn" href="#demo">Try the demo</a>' if demo else ""}
<a class="btn" href="{REPO_URL}">View on GitHub</a></div>
{_code(INSTALL)}
{"" if simulated else _stats(results)}
{banner}
</div>

<section class="block" id="how"><div class="wrap">
<h2>How it works</h2>
<p class="sub">Brink measures one thing: how strongly the model prefers to refuse a fixed set of harmful probe
prompts, compared with the model before fine-tuning. It uses that measure at three points.</p>
<div class="steps">
<div class="step"><span class="when">Before training</span><h3>Gate</h3>
<p>Scores every training sample by how much one training step on it would erode refusal, and drops the
riskiest. You get a ranked list and a cleaned file.</p></div>
<div class="step"><span class="when">During training</span><h3>Monitor</h3>
<p>Probes the model every few steps with one forward pass and raises an alarm when refusal stays below a
threshold calibrated on safe fine-tuning.</p></div>
<div class="step"><span class="when">On alarm</span><h3>Repair</h3>
<p>Halts the run, or rolls back to the last safe checkpoint and constrains the following updates so they
cannot lower refusal.</p></div>
<div class="step"><span class="when">After training</span><h3>Verify</h3>
<p>Checks the result on held-out harmful prompts, harmless prompts and task data, and writes a report.</p></div>
</div>
</div></section>

<section class="block" id="use"><div class="wrap">
<h2>Use it</h2>
<p class="sub">Python 3.10 or later. Brink loads AdvBench and HarmBench as probe prompts, which are gated on
Hugging Face: accept their terms once and log in with <code>huggingface-cli login</code>.</p>

<div class="use"><h3>1. Clean your data before you train</h3>
<p>Point Brink at a fine-tuning file (<code>.jsonl</code>, <code>.json</code> or <code>.csv</code> with prompt
and response) and the model you will fine-tune.</p>{_code(GATE_CLI)}</div>

<div class="use"><h3>2. Protect a training run</h3>
<p>Add one callback to a Hugging Face <code>Trainer</code> or TRL <code>SFTTrainer</code>. Brink probes the model
as it trains and halts when the alarm fires. Use <code>"rollback_guard"</code> to roll back and keep training
under the guard instead.</p>{_code(CALLBACK)}</div>

<div class="use"><h3>3. Inside a CuratorKIT pipeline</h3>
<p>The gate is a native CuratorKIT <code>BaseGate</code>, so it drops into an existing curation pipeline.
With AlignTune, attach the same callback with <code>attach_to_aligntune(trainer, callback)</code>.</p>
{_code(CURATOR)}</div>

<div class="use"><h3>4. See all of it on a model, in one command</h3>
<p>Runs the full loop on a small model, on CPU in about 20 minutes or on a GPU in a few, and writes a report
you can read.</p>{_code(RUN_CLI)}</div>

<div class="use"><h3>5. Start from a model Brink already protected</h3>
<p>A Hindi assistant fine-tuned from Tiny Aya on data with 10 % harmful samples, protected by Brink's gate,
alarm and rollback. Load the LoRA adapter on top of the base model.</p>{_code(ADAPTER)}</div>

<div class="use"><h3>6. Or use the app</h3>
<p>The same three jobs in a browser: run Brink live, upload a file to gate, and replay any run with your own
alarm rule.</p>{_code(APP)}</div>
</div></section>
{_demo(demo)}
<section class="block" id="limits"><div class="wrap">
<h2>What to know before you rely on it</h2>
<p class="sub">Measured limits, from the experiments below.</p>
<ul class="plain">
<li>The alarm fires when refusals break, not before. It is a tripwire, not a forecast.</li>
<li>The gate and halting at the alarm are the parts that hold up on their own. Rollback with the guard, used
alone on heavily contaminated data, was less safe than mixing in safety data.</li>
<li>Outside English the alarm needs its own calibration: with the English threshold it also fired on safe
fine-tuning in Hindi and Tagalog.</li>
<li>Tested on one model family (Tiny Aya, 3.35B) with one or two seeds.</li>
<li>This page runs no model. Gating your data and protecting a run happen on your machine.</li>
</ul>
</div></section>

<section class="block" id="evidence"><div class="wrap">
<h2>Evidence</h2>
<p class="sub">Every experiment's report as the run wrote it, with its pre-set pass criteria, plots and logs.
The "AuditKIT attacks resisted" column in the repair report predates a scoring fix: that run used AuditKIT's
default detectors, which count a refusal as a failed attack. Read its attack success rate (ASR) instead.</p>
{_evidence(results, stages)}
</div></section>

<section class="block" id="resources"><div class="wrap">
<h2>Resources</h2>
<p class="sub">Everything is open: code, the protected adapter, and every log behind the numbers above.</p>
<div class="links">
<a href="{REPO_URL}"><b>Source code</b><span>GitHub: library, CLI, app, experiments and tests</span></a>
<a href="https://huggingface.co/{ADAPTER_REPO}"><b>Protected adapter</b><span>Hugging Face model repo, with the
fingerprints</span></a>
<a href="https://huggingface.co/datasets/{RESULTS_REPO}"><b>Logs and reports</b><span>Hugging Face dataset: every
run's signals, evaluations and reports</span></a>
<a href="{REPO_URL}/blob/main/docs/ARTICLE.md"><b>Write-up</b><span>Method, datasets, metrics, results and
limitations</span></a>
</div>
</div></section>
</main>

<footer><div class="wrap"><span>Brink · MIT licence</span><a href="{REPO_URL}">GitHub</a>
<a href="https://huggingface.co/{ADAPTER_REPO}">Adapter</a><a href="https://huggingface.co/datasets/{RESULTS_REPO}">Logs</a>
<a href="{SPACE_URL}">Hugging Face Space</a></div></footer>
{data_script}
<script>{SCRIPT}</script>
</body>
</html>
"""
