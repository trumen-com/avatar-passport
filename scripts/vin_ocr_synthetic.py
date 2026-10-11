"""Synthetic VIN OCR check: render random valid VINs on plate-like images and read them back with read_vin.

Usage: YOLO_AUTOINSTALL=false .venv/bin/python scripts/vin_ocr_synthetic.py
Seed 20261011, 60 VINs (WMIs from data/reference/wmi_table.csv), every second image "hard": tilt +-5 deg, blur,
brightness, noise and JPEG 55-90. Prints exact / wrong / not-read counts per group.
Clean synthetic text, not real plates: it shows whether the reader handles other characters, not field accuracy."""
import csv, io, json, random, sys, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from avtopassport.vin.ocr import read_vin
from avtopassport.vin.validate import validate

VAL = {**{str(d): d for d in range(10)}, **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMN", range(1, 6))),
       "P": 7, "R": 9, **dict(zip("STUVWXYZ", range(2, 10)))}
W = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]
ALPH = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
YEARS = "ABCDEFGHJKLMNPRSTVWXY123456789"
def make_vin(rng, wmi):
    v = list(wmi + "".join(rng.choice(ALPH) for _ in range(5)) + "0" + rng.choice(YEARS) + rng.choice(ALPH)
             + "".join(rng.choice("0123456789") for _ in range(6)))
    r = sum(VAL[c] * w for c, w in zip(v, W)) % 11
    v[8] = "X" if r == 10 else str(r)
    return "".join(v)
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
def render(rng, vin, hard):
    bg = rng.choice([(235, 235, 230), (205, 208, 212), (60, 62, 66)])
    fg = (20, 20, 20) if bg[0] > 128 else (225, 225, 220)
    img = Image.new("RGB", (1200, 420), bg)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(rng.choice(FONTS), rng.randint(54, 74))
    small = ImageFont.truetype(FONTS[1], 30)
    d.text((60, 40), rng.choice(["TOYOTA MOTOR CORPORATION", "MADE IN CHINA", "BYD AUTO CO., LTD", "GM UZBEKISTAN"]), font=small, fill=fg)
    d.text((60, 160), (rng.choice(["VIN ", "VIN: ", ""]) + vin), font=f, fill=fg)
    d.text((60, 300), f"{rng.randint(1000, 2400)} KG   {rng.randint(1, 12):02d}/20{rng.randint(15, 26)}", font=small, fill=fg)
    a = np.array(img).astype(np.float32)
    if hard:
        img = Image.fromarray(a.astype(np.uint8)).rotate(rng.uniform(-5, 5), expand=True, fillcolor=bg)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.6)))
        a = np.array(img).astype(np.float32)
        a = a * rng.uniform(0.7, 1.15) + rng.uniform(-25, 25) + np.random.default_rng(rng.randint(0, 10**6)).normal(0, 7, a.shape)
    buf = io.BytesIO(); Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=rng.randint(55, 90) if hard else 92)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
wmis = [r["wmi"] for r in csv.DictReader(open("data/reference/wmi_table.csv"))]
rng = random.Random(20261011)
res = {"easy": [], "hard": []}
t0 = time.time()
for i in range(60):
    hard = i % 2 == 1
    vin = make_vin(rng, rng.choice(wmis))
    assert validate(vin).check_digit_ok, vin
    out = read_vin(render(rng, vin, hard))
    got = out.best.text if out.best else None
    res["hard" if hard else "easy"].append(("exact" if out.status == "read" and got == vin else
                                            "wrong" if out.status == "read" else "not_read", vin, got, out.status))
for k, v in res.items():
    c = {s: sum(1 for x in v if x[0] == s) for s in ("exact", "wrong", "not_read")}
    print(k, len(v), c)
    for x in v:
        if x[0] != "exact": print("   ", x)
print(f"avg {(time.time() - t0) / 60:.2f} s per image")
