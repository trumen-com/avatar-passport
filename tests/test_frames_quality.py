"""frames + quality on synthetic clips and images only (CPU, offline, no model weights, no data/raw)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from avtopassport.pipeline.frames import (
    Frame,
    _rotation_from_matrix,
    _sample_indices,
    extract_frames,
    hlg_to_sdr,
    pq_to_sdr,
    probe_video,
)
from avtopassport.pipeline.quality import (
    DEFAULT_THRESHOLDS,
    QualityThresholds,
    assess,
    brightness,
    filter_frames,
    sharpness,
)

FPS = 10.0
N_FRAMES = 30
W, H = 360, 640  # portrait


def grey(i: int) -> int:
    """Frame i is filled with this grey level, so a decoded frame tells which source frame it is."""
    return 8 * i


@pytest.fixture(scope="module")
def portrait_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("video") / "portrait.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    assert vw.isOpened()
    for i in range(N_FRAMES):
        img = np.full((H, W, 3), grey(i), np.uint8)
        img[:80, :60] = (0, 0, 255)  # red marker in the top-left corner
        vw.write(img)
    vw.release()
    return p


def write_pyav_clip(path: Path, rgb: np.ndarray, *, n: int = 6, rotation: int = 0, hlg: bool = False) -> Path:
    """Tiny MPEG-4 clip written with PyAV, optionally with a display matrix and HLG/BT.2020 tags."""
    av = pytest.importorskip("av")
    from av.video.reformatter import ColorPrimaries, ColorRange, Colorspace, ColorTrc

    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = rgb.shape[1], rgb.shape[0], "yuv420p"
        if hlg:
            cc = s.codec_context
            cc.color_trc, cc.color_primaries = ColorTrc.ARIB_STD_B67, ColorPrimaries.BT2020
            cc.colorspace, cc.color_range = Colorspace.BT2020, ColorRange.MPEG
        if rotation:
            s.set_display_rotation(rotation)
        for _ in range(n):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            if hlg:
                frame = frame.reformat(
                    format="yuv420p", dst_colorspace=Colorspace.BT2020, dst_color_range=ColorRange.MPEG
                )
            for pkt in s.encode(frame):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


# --------------------------------------------------------------------------- frames


@pytest.mark.parametrize("backend", ["opencv", "pyav"])
def test_extract_portrait_clip_at_2_5_fps(portrait_clip: Path, backend: str) -> None:
    frames, info, errors = extract_frames(portrait_clip, target_fps=2.5, backend=backend)
    assert errors == []
    assert info.backend == backend
    assert (info.width, info.height, info.frame_count, info.rotation, info.hdr) == (W, H, N_FRAMES, 0, False)
    assert info.fps == pytest.approx(FPS) and info.duration_s == pytest.approx(3.0)
    assert [f.index for f in frames] == [0, 4, 8, 12, 16, 20, 24, 28]
    assert [f.t_s for f in frames] == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])
    for f in frames:
        assert f.image.shape == (H, W, 3) and f.image.dtype == np.uint8 and f.source == str(portrait_clip)
        assert abs(f.image[200:, 100:].mean() - grey(f.index)) < 6  # the right source frame was decoded
        b, g, r = f.image[20, 20]
        assert r > 180 and b < 80  # marker still top-left: not rotated or mirrored


def test_backends_agree_and_auto_picks_opencv_for_sdr(portrait_clip: Path) -> None:
    a, info_auto, _ = extract_frames(portrait_clip)
    b, _, _ = extract_frames(portrait_clip, backend="pyav")
    assert info_auto.backend == "opencv"
    assert [f.index for f in a] == [f.index for f in b]
    for fa, fb in zip(a, b):
        assert np.abs(fa.image.astype(int) - fb.image.astype(int)).mean() < 2


def test_max_frames_spans_the_clip_and_max_side(portrait_clip: Path) -> None:
    frames, _, errors = extract_frames(portrait_clip, target_fps=2.5, max_frames=3, max_side=320)
    assert errors == []
    assert [f.index for f in frames] == [0, 16, 28]
    assert all(f.image.shape == (320, 180, 3) for f in frames)


def test_sample_indices() -> None:
    idx = _sample_indices(861, 30000 / 1001, 2.5, None)  # 29.97 fps walkaround metadata
    assert len(idx) == 72 and idx[:6] == [0, 12, 24, 36, 48, 60] and idx[-1] == 851
    assert set(np.diff(idx).tolist()) <= {11, 12}
    assert _sample_indices(30, 10.0, 30.0, None) == list(range(30))  # target above source fps: every frame
    assert _sample_indices(0, 10.0, 2.5, None) == []
    # a header claiming 2^40 frames: the max_frames subset is computed directly (it used to build the whole grid)
    huge = _sample_indices(2**40, 30.0, 2.5, 100)
    assert len(huge) == 100 and huge[0] == 0 and all(np.diff(huge) > 0) and huge[-1] < 2**40


@pytest.mark.parametrize("backend", ["opencv", "pyav"])
def test_display_rotation_is_applied(tmp_path: Path, backend: str) -> None:
    coded = np.full((32, 64, 3), 40, np.uint8)  # landscape as coded, RGB
    coded[:12, :16] = (255, 0, 0)              # red in the coded top-left
    clip = write_pyav_clip(tmp_path / "rotated.mp4", coded, rotation=-90)  # what an iPhone portrait clip carries
    frames, info, errors = extract_frames(clip, target_fps=5, backend=backend)
    assert errors == [] and info.rotation == -90 and (info.width, info.height) == (32, 64)
    img = frames[0].image
    assert img.shape == (64, 32, 3)
    b, g, r = img[3, -4]   # -90 = 90 deg clockwise: coded top-left -> display top-right
    assert r > 160 and b < 90


def test_hlg_clip_uses_pyav_tone_mapping_and_rotation(tmp_path: Path) -> None:
    coded = np.full((48, 64, 3), 128, np.uint8)
    coded[:12, :16] = (255, 0, 0)  # red in the coded top-left
    clip = write_pyav_clip(tmp_path / "hlg.mp4", coded, hlg=True, rotation=-90)
    info = probe_video(clip)
    assert info.hdr and info.color_trc == "arib-std-b67" and info.color_primaries == "bt2020"
    assert (info.width, info.height, info.rotation) == (48, 64, -90)

    frames, info, errors = extract_frames(clip, target_fps=5)
    assert errors == [] and info.backend == "pyav" and frames
    img = frames[0].image
    assert img.shape == (64, 48, 3)
    expected = hlg_to_sdr(np.full((1, 1, 3), 128 / 255, np.float32))[0, 0].astype(int)
    assert np.abs(img[30:, :30].reshape(-1, 3).mean(0) - expected).max() < 4  # tone-mapped grey
    b, g, r = img[3, -4]   # marker at the display top-right
    assert r > 150 and b < 90

    small, _, errors = extract_frames(clip, target_fps=5, max_side=32)  # swscale downscale before tone mapping
    assert errors == [] and small[0].image.shape == (32, 24, 3)
    assert np.abs(small[0].image[16:, :12].reshape(-1, 3).mean(0) - expected).max() < 4


def test_rotation_from_display_matrix() -> None:
    def matrix(a: int, b: int, c: int, d: int) -> bytes:  # 16.16 fixed point, FFmpeg layout
        return np.array([a, b, 0, c, d, 0, 0, 0, 1 << 30], "<i4").tobytes()

    one = 1 << 16
    assert _rotation_from_matrix(matrix(one, 0, 0, one)) == 0
    assert _rotation_from_matrix(matrix(0, one, -one, 0)) == -90  # as written for an iPhone portrait clip
    assert _rotation_from_matrix(matrix(0, -one, one, 0)) == 90
    assert _rotation_from_matrix(matrix(-one, 0, 0, -one)) == 180


def write_intra_clip(path: Path, n: int = 40) -> Path:
    """Intra-only MPEG-4 clip, index first (faststart): frame i has a flat left half of grey 6*i and a noise
    right half (packets of a few hundred bytes), and every frame decodes on its own."""
    av = pytest.importorskip("av")
    with av.open(str(path), "w", options={"movflags": "faststart"}) as c:
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt = 96, 64, "yuv420p"
        s.codec_context.gop_size, s.bit_rate = 1, 8_000_000  # high rate: the grey survives within 1 level
        for i in range(n):
            img = np.full((64, 96, 3), 6 * i, np.uint8)
            img[:, 48:] = np.random.default_rng(i).integers(0, 256, (64, 48, 3), dtype=np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


def corrupt_packet(src: Path, dst: Path, k: int) -> Path:
    """Copy of src with the payload of video packet k overwritten; the container index stays intact."""
    import av

    with av.open(str(src)) as c:
        pos, size = [(pkt.pos, pkt.size) for pkt in c.demux(c.streams.video[0]) if pkt.size][k]
    data = bytearray(src.read_bytes())
    data[pos:pos + size] = b"\xff" * size
    dst.write_bytes(bytes(data))
    return dst


@pytest.mark.parametrize("k", [5, 37])  # 37: inside the frame-threading flush window on a many-core CPU
def test_a_corrupt_packet_costs_only_its_own_frame(tmp_path: Path, k: int) -> None:
    # Regression: OpenCV stops reading at the first corrupt frame (auto mode returned frames 0..k-1 only),
    # and PyAV with frame threading lost the frames still queued in the decoder after the error.
    clip = corrupt_packet(write_intra_clip(tmp_path / "clean.mp4"), tmp_path / "bad.mp4", k)
    for backend in ("auto", "pyav"):
        frames, info, errors = extract_frames(clip, target_fps=10, backend=backend)
        assert [f.index for f in frames] == [i for i in range(40) if i != k]
        assert info.backend == "pyav" and errors
        for f in frames:  # indices stay aligned with the source after the lost packet
            assert abs(f.image[16:48, 8:24].mean() - 6 * f.index) < 3  # half the step between frames


def test_truncated_clips(tmp_path: Path, portrait_clip: Path) -> None:
    # index at the end (OpenCV's writer): a cut file has no index and cannot be opened at all
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(portrait_clip.read_bytes()[: portrait_clip.stat().st_size // 2])
    with pytest.raises(ValueError):
        extract_frames(cut)
    # index first (faststart): the frames before the cut come back, the rest are reported, nothing raises
    data = write_intra_clip(tmp_path / "fs.mp4").read_bytes()
    cut.write_bytes(data[: len(data) // 2])
    for backend in ("auto", "opencv", "pyav"):
        frames, _, errors = extract_frames(cut, target_fps=10, backend=backend)
        assert 10 <= len(frames) < 40 and [f.index for f in frames] == list(range(len(frames)))
        assert any("not decoded" in e for e in errors)


def test_extract_frames_bad_inputs(tmp_path: Path, portrait_clip: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_frames(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 1000)
    with pytest.raises(ValueError):
        extract_frames(junk)
    with pytest.raises(ValueError):
        extract_frames(portrait_clip, backend="ffmpeg")
    with pytest.raises(ValueError):
        extract_frames(portrait_clip, target_fps=0)


# --------------------------------------------------------------------------- HDR -> SDR


def test_hlg_to_sdr_levels_and_layout() -> None:
    black = hlg_to_sdr(np.zeros((4, 6, 3), np.uint16))
    assert black.dtype == np.uint8 and black.shape == (4, 6, 3) and black.max() == 0
    # HLG 0.5 -> E = 0.5^2/3 = 0.0833 -> OOTF E^1.2 = 0.0507 -> / 0.2032 (75 % HLG) = 0.2496
    # -> sRGB 1.055 * 0.2496^(1/2.4) - 0.055 = 0.537 -> 137 (below the soft-clip knee)
    mid = hlg_to_sdr(np.full((2, 2, 3), 0.5, np.float32))
    assert abs(int(mid[0, 0, 0]) - 137) <= 1 and (mid == mid[0, 0, 0]).all()
    u16, unit = np.full((2, 2, 3), 32768, np.uint16), np.full((2, 2, 3), 32768 / 65535)
    assert np.array_equal(hlg_to_sdr(u16), hlg_to_sdr(unit))  # 16-bit lookup table == float formula
    ramp = hlg_to_sdr(np.repeat(np.linspace(0, 1, 50, dtype=np.float32)[None, :, None], 3, axis=2))[0, :, 0]
    assert (np.diff(ramp.astype(int)) >= 0).all() and ramp[-1] >= 245
    red = hlg_to_sdr(np.array([[[0.75, 0.0, 0.0]]], np.float32))[0, 0]  # RGB in, BGR out
    assert red[2] > 200 and red[0] == 0 and red[1] == 0


def test_pq_to_sdr_levels() -> None:
    assert pq_to_sdr(np.zeros((2, 2, 3), np.uint16)).max() == 0
    ramp = pq_to_sdr(np.repeat(np.linspace(0, 1, 50, dtype=np.float32)[None, :, None], 3, axis=2))[0, :, 0]
    assert (np.diff(ramp.astype(int)) >= 0).all()
    white = pq_to_sdr(np.full((1, 1, 3), 0.5807, np.float32))[0, 0, 0]  # PQ code value of ~203 cd/m2
    assert 235 <= int(white) <= 250


# --------------------------------------------------------------------------- quality


def checkerboard(h: int = 480, w: int = 640, cell: int = 8) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return cv2.cvtColor((((y // cell + x // cell) % 2) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def test_quality_sharp_blurred_dark_bright() -> None:
    sharp = checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    dark = sharp // 12            # 0 / 21
    bright = np.full_like(sharp, 250)
    assert sharpness(sharp) > 100 * max(sharpness(blurred), 1.0)
    assert brightness(sharp) == pytest.approx(127.5, abs=1)
    ok = assess(sharp)
    assert ok.ok and ok.reason is None and ok.sharpness >= DEFAULT_THRESHOLDS.min_sharpness
    assert assess(blurred).reason == "blurry" and not assess(blurred).ok
    assert assess(dark).reason == "too_dark"   # exposure is reported before sharpness
    assert assess(bright).reason == "too_bright"
    assert assess(sharp, QualityThresholds(min_sharpness=1e12)).reason == "blurry"


def test_quality_is_measured_at_analysis_size() -> None:
    sharp = checkerboard()
    big = cv2.resize(sharp, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)  # 2560 px long side
    assert sharpness(big) == pytest.approx(sharpness(sharp), rel=0.05)
    assert brightness(big) == pytest.approx(brightness(sharp), abs=1)


def test_filter_frames_keeps_order_and_reasons() -> None:
    images = [checkerboard(), cv2.GaussianBlur(checkerboard(), (0, 0), 6), checkerboard() // 12, checkerboard()]
    frames = [Frame(i, i / 2.5, img, "synthetic") for i, img in enumerate(images)]
    kept, rejected, results = filter_frames(frames)
    assert [f.index for f in kept] == [0, 3]
    assert [(f.index, r.reason) for f, r in rejected] == [(1, "blurry"), (2, "too_dark")]
    assert [r.ok for r in results] == [True, False, False, True]


def test_filter_frames_rejects_unreadable_image_instead_of_raising() -> None:
    kept, rejected, results = filter_frames([Frame(0, 0.0, np.zeros((4, 4, 3), np.float32), "synthetic")])
    assert kept == [] and len(rejected) == 1 and results[0].reason == "invalid"
