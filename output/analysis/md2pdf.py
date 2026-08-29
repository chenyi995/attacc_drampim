#!/usr/bin/env python3
"""RESULTS_k*.md -> RESULTS_k*.pdf  (Chinese via Noto CJK, compact wide tables).
    python3 md2pdf.py
"""
import re, sys, markdown, weasyprint

def inline_math(s):  # tiny latex -> html (only $\max_i C_i$ appears)
    s = s.replace(r"\max", "max").replace(r"\min", "min")
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9])", r"<sup>\1</sup>", s)
    return s.replace("\\", "")

CSS = """
@page { size: A4; margin: 1.4cm 1.2cm; @bottom-center { content: counter(page);
        font: 8pt 'Noto Sans CJK SC'; color:#888; } }
body { font-family:'Noto Sans CJK SC','Noto Sans',sans-serif; font-size:9.5pt;
       line-height:1.45; color:#1a1a1a; }
h1 { font-size:17pt; border-bottom:2px solid #2166AC; padding-bottom:3px;
     color:#153; }
h2 { font-size:13pt; margin-top:14px; border-bottom:1px solid #ccc;
     padding-bottom:2px; color:#2166AC; }
h3 { font-size:11pt; margin-top:11px; color:#333; }
code { font-family:'Noto Sans Mono CJK SC',monospace; font-size:8.5pt;
       background:#f2f2ef; padding:0 3px; border-radius:3px; }
table { border-collapse:collapse; width:100%; margin:6px 0; font-size:7.6pt;
        table-layout:auto; }
th,td { border:1px solid #cfcfc7; padding:2px 4px; text-align:right;
        word-break:break-word; }
th { background:#eef2f7; color:#153; text-align:center; }
td:first-child, th:first-child, td:nth-child(3) { text-align:left; }
tr:nth-child(even) td { background:#fafaf8; }
blockquote { border-left:3px solid #e0a500; margin:6px 0; padding:2px 10px;
             color:#555; background:#fffdf5; font-size:9pt; }
strong { color:#111; }
"""

MD_EXT = ["tables", "fenced_code", "sane_lists"]
for k in ("k2", "k8", "k32"):
    raw = open(f"RESULTS_{k}.md").read()
    raw = re.sub(r"\$([^$]+)\$", lambda m: inline_math(m.group(1)), raw)
    body = markdown.markdown(raw, extensions=MD_EXT)
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>" \
           f"<body>{body}</body></html>"
    weasyprint.HTML(string=html).write_pdf(f"RESULTS_{k}.pdf")
    print(f"wrote RESULTS_{k}.pdf")
