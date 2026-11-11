#!/usr/bin/env python3
"""Render README.md into an identical, print-formatted Korean Portfolio PDF via Chrome."""

from __future__ import annotations

import argparse
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

CHROME_MAC_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def render_pdf(readme_path: str = "README.md", out_pdf: str = "포트폴리오_README.pdf") -> None:
    md_file = Path(readme_path).resolve()
    base_dir = md_file.parent
    md_text = md_file.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "nl2br"],
    )

    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <base href="{base_dir.as_uri()}/">
  <title>분리비전 AI (Bunri-Vision) 포트폴리오</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
  <style>
    @page {{
      size: A4;
      margin: 15mm 14mm 16mm 14mm;
    }}
    body {{
      font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", sans-serif;
      color: #18221d;
      line-height: 1.62;
      font-size: 10.2pt;
      margin: 0;
    }}
    h1 {{
      font-size: 19pt;
      border-bottom: 2.5px solid #0c7a53;
      padding-bottom: 8px;
      margin-top: 0;
      color: #0b3b2b;
    }}
    h2 {{
      font-size: 14pt;
      color: #0b3b2b;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 22px;
      page-break-after: avoid;
    }}
    h3 {{
      font-size: 11.4pt;
      color: #1e293b;
      margin-top: 15px;
      page-break-after: avoid;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 14px 0;
      font-size: 9pt;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #ecfdf5;
      color: #064e3b;
      font-weight: 700;
    }}
    pre {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 9px 11px;
      font-size: 8.6pt;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      page-break-inside: avoid;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 4px;
      font-size: 8.8pt;
    }}
    pre code {{
      background: none;
      padding: 0;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1px solid #e2e8f0;
      margin: 8px 0;
      page-break-inside: avoid;
    }}
    blockquote {{
      margin: 10px 0;
      padding: 8px 12px;
      border-left: 4px solid #0c7a53;
      background: #f0fdf4;
      color: #14532d;
    }}
    a {{
      color: #0c7a53;
      text-decoration: none;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>"""

    tmp_html = base_dir / "_tmp_portfolio_render.html"
    tmp_html.write_text(full_html, encoding="utf-8")

    try:
        with sync_playwright() as p:
            launch_kwargs = {"headless": True}
            if Path(CHROME_MAC_PATH).exists():
                launch_kwargs["executable_path"] = CHROME_MAC_PATH
            browser = p.chromium.launch(**launch_kwargs)
            page = browser.new_page()
            page.goto(tmp_html.as_uri(), wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(700)
            page.pdf(
                path=str(Path(out_pdf).resolve()),
                format="A4",
                print_background=True,
                display_header_footer=False,
            )
            browser.close()
        print(f"[OK] Rendered PDF: {out_pdf} ({Path(out_pdf).stat().st_size / (1024**2):.2f} MB)")
    finally:
        if tmp_html.exists():
            tmp_html.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--out", default="포트폴리오_README.pdf")
    args = parser.parse_args()
    render_pdf(args.readme, args.out)
