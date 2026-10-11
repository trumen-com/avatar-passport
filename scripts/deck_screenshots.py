"""Deck screenshots of a finished passport page, with the VIN masked in the page before capture.

Usage (app running on localhost:8000):
    .venv/bin/python scripts/deck_screenshots.py <job id> docs/assets/screenshots --width 430
Opens each passport tab (car, 3D model, customize, damage, body shops, VIN / history, report) and captures the
viewport at each section.
Every VIN-like string on the page (17 chars of the VIN alphabet) is replaced by its first 11 characters + "******"
before any capture, so the committed images never contain a full VIN. The page shows only blurred frames.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screenshot import Marionette  # noqa: E402

MASK_JS = r"""
const re = /\b([A-HJ-NPR-Z0-9]{11})[A-HJ-NPR-Z0-9]{6}\b/g;
const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
let n = 0;
while (walker.nextNode()) {
  const t = walker.currentNode;
  const v = t.nodeValue.replace(re, (m, p) => { n++; return p + "******"; });
  if (v !== t.nodeValue) t.nodeValue = v;
}
for (const i of document.querySelectorAll("input")) { if (re.test(i.value)) { i.value = ""; n++; } }
return n;
"""

SECTIONS = [  # (file name, tab, CSS selector to scroll to; None = top of the page)
    ("00_car", "overview", None),
    ("01_summary", "overview", "#summary-title"),
    ("02_damage", "defects", None),
    ("03_3d_hq", "model3d", None),
    ("04_3d_cloud", "model3d", "#recon-card"),
    ("05_customize", "tryon", None),
    ("06_shops", "shops", None),
    ("07_vin_history", "vin", None),
    ("08_diff", "vin", "#diff-card"),
    ("09_report", "report", None),
]

SHOW_JS = """
location.hash = arguments[0];
const sel = arguments[1];
if (!sel) { window.scrollTo(0, 0); return true; }
const e = document.querySelector(sel);
if (!e || e.hidden) return false;
e.style.scrollMarginTop = getComputedStyle(document.querySelector('.panels')).scrollMarginTop;
e.scrollIntoView({block: 'start'});
return true;
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_id")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--width", type=int, default=430)
    ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args(argv)
    profile = Path(tempfile.mkdtemp(prefix="ffshot-", dir=Path.home()))
    proc = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(profile)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        m = Marionette(2828)
        m.cmd("WebDriver:NewSession", {"capabilities": {}})
        m.cmd("WebDriver:SetWindowRect", {"width": a.width, "height": a.height})
        m.cmd("WebDriver:Navigate", {"url": f"{a.base}/?inspection={a.job_id}"})
        for _ in range(90):
            if m.cmd("WebDriver:ExecuteScript", {"script": "return !!document.querySelector('#vin-value') && "
                                                 "document.querySelector('#vin-value').textContent.length > 0;",
                                                 "args": []})["value"]:
                break
            time.sleep(1)
        time.sleep(3)
        masked = m.cmd("WebDriver:ExecuteScript", {"script": MASK_JS, "args": []})["value"]
        print(f"masked {masked} VIN strings")
        seen_3d = False
        for name, tab, sel in SECTIONS:
            ok = m.cmd("WebDriver:ExecuteScript", {"script": SHOW_JS, "args": [tab, sel]})["value"]
            if not ok:
                print(f"skip {name}: {sel} not shown")
                continue
            if tab == "model3d" and not seen_3d:
                seen_3d = True
                time.sleep(25)  # the 3D models are loaded when their tab is first opened (point cloud + 28 MB splat)
                m.cmd("WebDriver:ExecuteScript", {"script": SHOW_JS, "args": [tab, sel]})
            time.sleep(2.5)
            m.cmd("WebDriver:ExecuteScript", {"script": MASK_JS, "args": []})
            png = m.cmd("WebDriver:TakeScreenshot", {"hash": False, "full": False})["value"]
            (a.out_dir / f"{name}.png").write_bytes(base64.b64decode(png))
            print(f"wrote {name}.png")
        m.cmd("WebDriver:DeleteSession")
    finally:
        proc.terminate()
        subprocess.run(["rm", "-rf", str(profile)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
