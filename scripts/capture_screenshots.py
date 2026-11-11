#!/usr/bin/env python3
"""Capture 5 high-DPI verification screenshots directly from the live deployed service URL."""

from __future__ import annotations

import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

CHROME_MAC_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def capture(base_url: str, out_dir: str = "assets/screenshots") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if Path(CHROME_MAC_PATH).exists():
            launch_kwargs["executable_path"] = CHROME_MAC_PATH
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
        page = context.new_page()

        print(f"[NAV] Opening {base_url} ...")
        page.goto(base_url, wait_until="networkidle", timeout=60000)
        page.wait_for_selector(".sample-btn", timeout=20000)
        page.wait_for_timeout(600)

        # Screenshot 1: Main overview with 9 held-out test samples ready
        page.screenshot(path=str(out / "screen1_main_overview.png"), full_page=False)
        print("[OK] Saved screen1_main_overview.png")

        # Screenshot 2: Click Plastic sample -> shows 3-view (raw / WB+CLAHE / Spatial CAM) + 3-step guide + 9-class bars
        plastic_btn = page.locator(".sample-btn").nth(6)  # Plastic
        plastic_btn.click()
        page.wait_for_selector("#prediction-content:not([hidden])", timeout=20000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(out / "screen2_prediction_cam.png"), full_page=True)
        print("[OK] Saved screen2_prediction_cam.png")

        # Screenshot 3: Click Vegetation sample -> shows high-impact Korean sorting penalty trap warning!
        veg_btn = page.locator(".sample-btn").nth(8)  # Vegetation
        veg_btn.click()
        page.wait_for_timeout(900)
        page.screenshot(path=str(out / "screen3_vegetation_alert.png"), full_page=True)
        print("[OK] Saved screen3_vegetation_alert.png")

        # Screenshot 4: Trigger Invalid Input Exception Guardrail (.txt non-image rejection)
        exc_btn = page.locator('[data-exception="non_image_txt"]')
        exc_btn.click()
        page.wait_for_selector("#error-banner:not([hidden])", timeout=15000)
        page.wait_for_timeout(500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / "screen4_exception_handling.png"), full_page=False)
        print("[OK] Saved screen4_exception_handling.png")

        # Screenshot 5: Switch to Tab 2 (Model Benchmark & Per-Class Performance Table)
        page.locator("#tab-benchmark").click()
        page.wait_for_timeout(600)
        page.screenshot(path=str(out / "screen5_benchmark_dashboard.png"), full_page=True)
        print("[OK] Saved screen5_benchmark_dashboard.png")

        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Live service URL to verify and screenshot")
    parser.add_argument("--out-dir", default="assets/screenshots")
    args = parser.parse_args()
    capture(args.url, args.out_dir)
