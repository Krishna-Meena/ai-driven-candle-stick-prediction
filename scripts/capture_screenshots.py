"""Playwright screenshot automation for the Streamlit dashboard.

Usage:
    uv run python scripts/capture_screenshots.py

Before first run, install Playwright browsers:
    uv run playwright install chromium
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = REPO_ROOT / "docs" / "screenshots"
PORT = 8502
BASE_URL = f"http://localhost:{PORT}"

PAGES = [
    ("Home", "home.png"),
    ("Market Overview", "market-overview.png"),
    ("Predictions", "predictions.png"),
    ("Training Center", "training-center.png"),
    ("Backtesting", "backtesting.png"),
    ("Model Comparison", "model-comparison.png"),
    ("About", "about.png"),
]


def _nav_click(page: Any, title: str) -> bool:
    """Try multiple selectors to find and click a sidebar nav link."""
    selectors = [
        f'a:has-text("{title}")',
        f'button:has-text("{title}")',
        f'span:has-text("{title}")',
        f'div:has-text("{title}")',
        f'[data-testid="stSidebar"] a:has-text("{title}")',
        f'[data-testid="stSidebar"] span:has-text("{title}")',
        f'aside section a:has-text("{title}")',
        # Streamlit 1.35+ uses stPageLink or st.navigation
        f'a[href*="{title.lower().replace(" ", "-")}"]',
    ]

    for selector in selectors:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=2000):
                el.click(timeout=5000)
                return True
        except Exception:
            continue
    return False


def _wait_for_stable(page: Any, extra_sleep: float = 2.0) -> None:
    """Wait for the page to finish rendering."""
    with contextlib.suppress(Exception):
        page.wait_for_load_state("networkidle", timeout=15000)
    with contextlib.suppress(Exception):
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    if extra_sleep:
        time.sleep(extra_sleep)


def capture() -> None:
    import playwright.sync_api as pw

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Start Streamlit server ─────────────────────────────────────────
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(REPO_ROOT / "src/ai_candle_predictor/presentation/dashboard/app.py"),
            f"--server.port={PORT}",
            "--server.headless=true",
            "--server.runOnSave=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=REPO_ROOT,
    )

    try:
        print(f"Waiting for Streamlit at {BASE_URL} ...", flush=True)
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=2,
            )
            page = context.new_page()

            # ── Retry connecting ────────────────────────────────────────
            connected = False
            for attempt in range(30):
                try:
                    page.goto(BASE_URL, timeout=10000)
                    _wait_for_stable(page, extra_sleep=1.0)
                    connected = True
                    break
                except Exception:
                    if attempt < 29:
                        time.sleep(2)
            if not connected:
                raise RuntimeError(f"Could not reach {BASE_URL} after 60s.")

            print(f"  Connected. Title: {page.title()}", flush=True)

            # ── Navigate and screenshot ─────────────────────────────────
            for title, filename in PAGES:
                filepath = SCREENSHOT_DIR / filename
                print(f"  [{title}] ", end="", flush=True)

                # Strategy 1: click sidebar nav link
                clicked = _nav_click(page, title)
                if clicked:
                    print("nav_click ", end="", flush=True)
                else:
                    # Strategy 2: try query param with url_path
                    page_id = f"page_{title.lower().replace(' ', '_')}"
                    if title == "Home":
                        page_id = "page_home"
                    elif title == "Model Comparison":
                        page_id = "page_model_comparison"
                    elif title == "Training Center":
                        page_id = "page_training_center"
                    elif title == "Market Overview":
                        page_id = "page_market_overview"
                    try:
                        page.goto(f"{BASE_URL}/?page={page_id}", timeout=10000)
                        print("URL_nav ", end="", flush=True)
                    except Exception:
                        print("fallback ", end="", flush=True)

                _wait_for_stable(page, extra_sleep=2.0)

                with contextlib.suppress(Exception):
                    page.wait_for_selector("h1, h2, h3, .main-header", timeout=5000)

                # Small scroll to trigger lazy renders
                page.evaluate("window.scrollTo(0, 200)")
                time.sleep(0.5)
                page.evaluate("window.scrollTo(0, 0)")
                time.sleep(0.5)

                page.screenshot(path=str(filepath), full_page=True)
                size_kb = filepath.stat().st_size / 1024
                print(f"{size_kb:.0f} KB", flush=True)

            browser.close()
            print("All screenshots captured.", flush=True)

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    capture()
