"""Render the finished experiments as one static HTML page (a Hugging Face Space with `sdk: static`).

Static Spaces are free and need no server, so the recorded results stay public even when the Gradio app can't
be hosted. The page shows every finished stage's REPORT.md and plots, and links to the CSV / JSON logs next to
it. It runs nothing: the live *Run Brink* and *Gate your data* tabs need the Gradio app (`make app`).

Only a small subset of Markdown is rendered, the subset the reports use: headings, paragraphs, bullet lists,
pipe tables, **bold**, `code` and [links](url). Everything is HTML-escaped first.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

REPO_URL = "https://github.com/andrew1234-arch/brink"
ORDER = ["live", "pipeline", "go_nogo", "gate", "repair", "multilingual", "circuit"]  # the product first
TITLES = {
    "live": "Brink end to end (recorded live run)",
    "pipeline": "Pipeline: a Hindi assistant, before and after",
    "go_nogo": "Go / no-go: is the drift detectable?",
    "gate": "Gate: finding the data that erodes safety",
    "repair": "Repair: rollback and the erosion guard vs baselines",
    "multilingual": "Multilingual: Hindi and Tagalog",
    "circuit": "CircuitKIT: the refusal circuit vs the monitored layer",
}

CSS = """
:root { --bg: #fbfbfa; --fg: #1c1e21; --muted: #5f6672; --line: #dcdfe4; --accent: #1d4ed8; --card: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14161a; --fg: #e7e9ec; --muted: #9aa2ae; --line: #2c313a; --accent: #8ab4ff; --card: #1b1e24; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.55 system-ui, sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.9rem; margin: 0 0 4px; }
h2 { font-size: 1.35rem; margin: 0 0 12px; }
h3, h4 { font-size: 1.05rem; margin: 20px 0 8px; }
a { color: var(--accent); }
.lead { color: var(--muted); margin: 0 0 16px; }
nav { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 24px; }
nav a { border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; text-decoration: none; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 20px;
  margin: 0 0 20px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 10px 0; font-size: 0.92rem; }
th, td { border: 1px solid var(--line); padding: 5px 9px; text-align: left; vertical-align: top; }
img { max-width: 100%; height: auto; background: #fff; border-radius: 6px; margin: 10px 0; }
code { font: 0.9em ui-monospace, monospace; background: var(--line); border-radius: 4px; padding: 1px 4px; }
.files { color: var(--muted); font-size: 0.9rem; }
"""


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
                else:  # a wrapped line of the item above
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


def _ordered(stages: list[str]) -> list[str]:
    return [s for s in ORDER if s in stages] + sorted(s for s in stages if s not in ORDER)


def render(results: Path, stages: list[str], simulated: bool = False) -> str:
    """The whole page. `results` is the folder published next to index.html as `results/`."""
    stages = _ordered(stages)
    nav = "".join(f'<a href="#{html.escape(s)}">{html.escape(s)}</a>' for s in stages)
    sections = []
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
        sections.append(
            f'<section id="{html.escape(s)}"><h2>{html.escape(TITLES.get(s, s))}</h2>'
            f"{markdown_to_html(report, shift=2)}{plots}<p class='files'>Files: {files}</p></section>"
        )
    roi = results / "ROI.md"
    if roi.is_file():
        sections.append(
            f'<section id="roi"><h2>Cost and ROI</h2>{markdown_to_html(roi.read_text(encoding="utf-8"), shift=2)}'
            "</section>"
        )
    banner = (
        "<section><strong>SIMULATED runs.</strong> No real experiment has finished yet; these are generated "
        "examples of the report format, not measurements.</section>"
        if simulated
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Brink: keep a model safe while you fine-tune it</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Brink</h1>
<p class="lead">Catches a language model's safety collapsing during fine-tuning, and stops it: a gate scores the
training data, a monitor watches the model's refusal while it trains, and on alarm a repair step rolls back and
guards the gradients. Measured on Tiny Aya (<code>CohereLabs/tiny-aya-global</code>) with the Lexsi toolkits.</p>
<p>These are the recorded results of every finished experiment, as the runs wrote them, including the criteria
that failed. Code, method and how to run the live app: <a href="{REPO_URL}">{REPO_URL}</a>.</p>
<p class="files">The "AuditKIT attacks resisted" columns in the pipeline and repair reports predate a scoring fix:
those runs used AuditKIT's default detectors, which count a refusal as a failed attack. Read the WildGuard-judged
attack success rate (ASR) instead.</p>
<nav>{nav}</nav>
{banner}
{"".join(sections)}
</main>
</body>
</html>
"""
