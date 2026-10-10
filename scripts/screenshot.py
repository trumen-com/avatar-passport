"""Full-page screenshots of the running web app with headless Firefox (Marionette protocol, no extra packages).

Usage (app running on localhost:8000):
    .venv/bin/python scripts/screenshot.py "http://localhost:8000/?inspection=<job id>" out.png --wait-for "#passport" --width 430

Firefox is started headless with --marionette on a private profile; the script navigates, waits until the CSS
selector exists and the page reports no pending loads (or the timeout passes), then saves a full-page PNG.
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class Marionette:
    def __init__(self, port: int = 2828, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.3)
        self.sock.settimeout(120)
        self.msg_id = 0
        self._read()  # server hello

    def _read(self) -> object:
        buf = b""
        while b":" not in buf:
            buf += self.sock.recv(1)
        length, rest = buf.split(b":", 1)
        need = int(length)
        data = rest
        while len(data) < need:
            data += self.sock.recv(need - len(data))
        return json.loads(data)

    def cmd(self, name: str, params: dict | None = None) -> object:
        self.msg_id += 1
        payload = json.dumps([0, self.msg_id, name, params or {}]).encode()
        self.sock.sendall(str(len(payload)).encode() + b":" + payload)
        while True:
            msg = self._read()
            if isinstance(msg, list) and msg[0] == 1 and msg[1] == self.msg_id:
                if msg[2]:
                    raise RuntimeError(f"{name}: {msg[2]}")
                return msg[3]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--wait-for", default="body", help="CSS selector that must exist (and be visible)")
    ap.add_argument("--wait-text", default=None, help="text that must appear in the page")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--settle", type=float, default=3.0, help="extra seconds after the condition is met")
    ap.add_argument("--width", type=int, default=430)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--port", type=int, default=2828)
    a = ap.parse_args(argv)
    profile = Path(tempfile.mkdtemp(prefix="ffshot-", dir=Path.home()))  # snap Firefox cannot read /tmp profiles
    proc = subprocess.Popen(["firefox", "--headless", "--marionette", "--no-remote", "--profile", str(profile),
                             "--remote-allow-system-access"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        m = Marionette(a.port)
        m.cmd("WebDriver:NewSession", {"capabilities": {"alwaysMatch": {"acceptInsecureCerts": True}}})
        m.cmd("WebDriver:SetWindowRect", {"width": a.width, "height": a.height})
        m.cmd("WebDriver:Navigate", {"url": a.url})
        deadline = time.time() + a.timeout
        script = ("const s = arguments[0], t = arguments[1]; const e = document.querySelector(s);"
                  "const ok = !!e && e.offsetParent !== null && (!t || document.body.innerText.includes(t));"
                  "return ok;")
        while time.time() < deadline:
            if m.cmd("WebDriver:ExecuteScript", {"script": script, "args": [a.wait_for, a.wait_text]}).get("value"):
                break
            time.sleep(1)
        else:
            print(f"timeout waiting for {a.wait_for!r} / {a.wait_text!r}; taking the screenshot anyway", file=sys.stderr)
        time.sleep(a.settle)
        png = m.cmd("WebDriver:TakeScreenshot", {"full": True, "hash": False})["value"]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_bytes(base64.b64decode(png))
        print(f"wrote {a.out}")
        m.cmd("WebDriver:DeleteSession")
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["rm", "-rf", str(profile)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
