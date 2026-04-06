"""Render an HTML page to a PNG with headless Chromium, for the images in docs/.

Playwright is not a dependency of the project: run this from a separate environment
(``pip install playwright pillow``) with a Chromium build installed, and pass the
browser with ``--chromium`` when Playwright's own download is not available. The PNG
is reduced to an adaptive 256-colour palette, which keeps flat report graphics sharp
at a fraction of the size.

Run:  python scripts/screenshot.py PAGE.html OUT.png [--width 1280] [--height 2000]
      [--dark] [--chromium /path/to/chrome]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main() -> None:
    """Screenshot the page (the top ``--height`` pixels, or all of it) and palettise it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("page", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--width", type=int, default=1280, help="viewport width (px)")
    parser.add_argument("--height", type=int, help="keep only the top of the page (px)")
    parser.add_argument("--dark", action="store_true", help="render the dark colour scheme")
    parser.add_argument("--chromium", help="Chromium executable to use")
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height or 900},
            color_scheme="dark" if args.dark else "light",
        )
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(args.page.resolve().as_uri())
        page.screenshot(path=str(args.out), full_page=args.height is None)
        browser.close()
    if errors:
        raise SystemExit(f"the page raised errors: {errors}")
    image = Image.open(args.out).convert("RGB")
    image.quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(args.out, optimize=True)
    print(f"{args.out}: {image.width}x{image.height}, {args.out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
