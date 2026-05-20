#!/usr/bin/env python3
"""Build a submission-ready PDF from FINAL_REPORT.md."""
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

HERE = Path(__file__).parent
md_path = HERE / "FINAL_REPORT.md"
pdf_path = HERE / "FINAL_REPORT.pdf"

md = md_path.read_text()

# Resolve image paths to absolute so the converter finds them
md = md.replace("](figures/", f"]({HERE / 'figures'}/")

pdf = MarkdownPdf(toc_level=2, optimize=True)

# CSS for readable academic style
css = """
@page { size: Letter; margin: 0.75in; }
body { font-family: 'DejaVu Sans', Helvetica, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin-top: 0.5em; }
h2 { font-size: 14pt; margin-top: 1em; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 12pt; margin-top: 0.8em; }
h4 { font-size: 11pt; margin-top: 0.6em; }
p { margin: 0.4em 0; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 9.5pt; }
th, td { border: 1px solid #888; padding: 4px 6px; vertical-align: top; }
th { background: #f0f0f0; font-weight: bold; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9.5pt; background: #f5f5f5; padding: 1px 3px; }
pre { background: #f5f5f5; padding: 6px 8px; font-size: 9pt; overflow-x: auto; border: 1px solid #ddd; }
img { max-width: 100%; height: auto; display: block; margin: 0.5em auto; }
blockquote { margin: 0.5em 1em; padding: 0.4em 0.8em; border-left: 3px solid #888; background: #fafafa; color: #333; }
"""

pdf.add_section(Section(md, toc=False), user_css=css)
pdf.meta["title"] = "Comparative Analysis of VLA Architectures: Pi0 vs Pi0-FAST"
pdf.meta["author"] = "Kartik Reddy Katam, Gaurav Dharmadhikari"
pdf.meta["subject"] = "CMPE 188 Final Project Report"

pdf.save(pdf_path)
print(f"Wrote {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
