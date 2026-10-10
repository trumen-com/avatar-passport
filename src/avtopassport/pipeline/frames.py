"""Frame extraction from walkaround videos.

- Default backend OpenCV (bundled FFmpeg; auto-rotates from metadata). PyAV is the fallback and the
  HDR path: if color_trc is HLG (arib-std-b67) or PQ (smpte2084), decode with PyAV to rgb48le and
  convert with hlg_to_sdr / pq_to_sdr, then apply rotation via np.rot90 from frame.rotation.
- Media facts (probe 2026-10-10): data/raw/video/360VIEW.mp4 = H.264 8-bit BT.709 1080x1920 29.97 fps;
  data/raw/originals/360VIEW.MOV = HEVC 10-bit HLG BT.2020, coded 1920x1080, PyAV rotation=-90,
  OpenCV returns (1920,1080,3) portrait.
- Never let one bad frame crash the run: skip it and record the error string. OpenCV stops at the first
  corrupt frame, so a short OpenCV read falls back to PyAV (which skips corrupt packets) in auto mode.
- Portrait is NOT asserted (landscape uploads are legal); orientation is recorded in VideoInfo.
- Timestamps are nominal: t_s = source frame index / fps (both backends count decoded frames; a packet
  that PyAV fails to decode single-threaded counts as one frame, so later indices stay aligned).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..io import resize_max_side

log = logging.getLogger("avtopassport.frames")

BACKENDS = ("auto", "opencv", "pyav")
HLG_TRC = "arib-std-b67"
PQ_TRC = "smpte2084"
_TRC_IDS = {HLG_TRC: 18, PQ_TRC: 16}  # FFmpeg AVColorTransferCharacteristic values
_UNSPECIFIED = (0, 2)                  # FFmpeg RESERVED0 / UNSPECIFIED colour enums

# ITU-R BT.2100 HLG inverse OETF constants (b = 1 - 4a, c = 0.5 - a*ln(4a)) and OOTF system gamma
_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991073
_HLG_GAMMA = 1.2
_BT2020_LUMA = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)
# linear-light BT.2020 -> BT.709 RGB, derived from the two sets of primaries (D65 white)
_BT2020_TO_BT709 = np.array(
    [[1.660491, -0.587641, -0.072850],
     [-0.124550, 1.132900, -0.008349],
     [-0.018151, -0.100579, 1.118730]],
    dtype=np.float32,
)
# SMPTE ST 2084 (PQ) EOTF constants
_PQ_M1, _PQ_M2 = 2610 / 16384, 2523 / 4096 * 128
_PQ_C1, _PQ_C2, _PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
# HDR reference white 203 cd/m2: reported (https://www.itu.int/pub/R-REP-BT.2408); used for PQ only.
PQ_REFERENCE_WHITE_NITS = 203.0
# Linear SDR level above which highlights are compressed (uncalibrated default).
SOFT_CLIP_KNEE = 0.8


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int                 # display orientation (after rotation)
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str
    rotation: int              # degrees as reported by the container (0 if none)
    pix_fmt: str
    color_trc: str | None
    color_primaries: str | None
    hdr: bool                  # HLG or PQ transfer detected
    backend: str               # "opencv" | "pyav"


@dataclass
class Frame:
    index: int                 # source frame index
    t_s: float                 # timestamp in seconds
    image: np.ndarray          # uint8 HxWx3 BGR, display orientation, SDR
    source: str                # video path


# --------------------------------------------------------------------------- probing


def _rate(x: Any) -> float:
    """Positive finite float or 0.0 (handles None, Fractions and 0/0 rates)."""
    try:
        v = float(x)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def _norm_deg(d: float) -> int:
    """Round to whole degrees in (-180, 180]."""
    r = int(round(d)) % 360
    return r - 360 if r > 180 else r


def _known(v: Any) -> bool:
    return v is not None and int(v) not in _UNSPECIFIED


def _color_name(enum_cls: Any, value: Any) -> str | None:
    """FFmpeg-style name of a colour enum value ('arib-std-b67', 'bt2020', ...); None if unspecified."""
    if not _known(value):
        return None
    try:
        return enum_cls(int(value)).name.lower().replace("_", "-")
    except ValueError:
        return str(int(value))


def _rotation_from_matrix(buf: bytes) -> int:
    """FFmpeg av_display_rotation_get() on a raw 3x3 int32 display matrix (16.16 fixed point)."""
    m = np.frombuffer(bytes(buf)[:36], dtype="<i4").astype(np.float64) / 65536.0
    s0, s1 = math.hypot(m[0], m[3]), math.hypot(m[1], m[4])
    if s0 == 0 or s1 == 0:
        return 0
    return _norm_deg(-math.degrees(math.atan2(m[1] / s1, m[0] / s0)))


def _fourcc(v: float) -> str:
    """OpenCV four-character code -> lower-case string ('' when unknown, i.e. <= 0)."""
    code = int(v)
    return code.to_bytes(4, "little").decode("ascii", "replace").strip("\x00 ").lower() if code > 0 else ""


def _probe_pyav(p: Path) -> VideoInfo:
    import av
    from av.video.reformatter import ColorPrimaries, ColorTrc

    with av.open(str(p)) as c:
        if not c.streams.video:
            raise ValueError("no video stream")
        s = c.streams.video[0]
        cc = s.codec_context
        codec = cc.name or "unknown"
        fps = _rate(s.average_rate) or _rate(s.guessed_rate) or _rate(s.base_rate)
        if s.duration and s.time_base:
            duration = float(s.duration * s.time_base)
        else:
            duration = c.duration / 1e6 if c.duration else 0.0
        frames = int(s.frames or 0)
        side = (getattr(cc, "coded_side_data", None) or {}).get("display_matrix")
        rotation = _rotation_from_matrix(side) if side else 0
        width, height, pix_fmt = cc.width, cc.height, cc.pix_fmt or "unknown"
        trc, prim = cc.color_trc, cc.color_primaries
        try:
            first = next(c.decode(s), None)
        except av.FFmpegError as exc:  # keep the container-level values
            log.warning("cannot decode the first frame of %s: %s", p, exc)
            first = None
        if first is not None:
            rotation = _norm_deg(first.rotation) if first.rotation else rotation
            width, height, pix_fmt = first.width, first.height, first.format.name
            trc = first.color_trc if _known(first.color_trc) else trc
            prim = first.color_primaries if _known(first.color_primaries) else prim
    if frames <= 0 and duration > 0 and fps > 0:
        frames = int(round(duration * fps))
    if duration <= 0 and frames > 0 and fps > 0:
        duration = frames / fps
    if rotation % 180:
        width, height = height, width
    trc_name = _color_name(ColorTrc, trc)
    return VideoInfo(
        path=str(p), width=width, height=height, fps=fps, frame_count=frames, duration_s=duration,
        codec=codec, rotation=rotation, pix_fmt=pix_fmt, color_trc=trc_name,
        color_primaries=_color_name(ColorPrimaries, prim), hdr=trc_name in _TRC_IDS, backend="pyav",
    )


def _probe_opencv(p: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV cannot open the file")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        fps = _rate(cap.get(cv2.CAP_PROP_FPS))
        frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # OpenCV reports the clockwise correction; the container (PyAV) convention is its negative
        rotation = _norm_deg(-cap.get(cv2.CAP_PROP_ORIENTATION_META))
        codec = _fourcc(cap.get(cv2.CAP_PROP_FOURCC)) or "unknown"
        pix_fmt = _fourcc(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)) or "unknown"
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise ValueError("no video stream")
    return VideoInfo(
        path=str(p), width=width, height=height, fps=fps, frame_count=frames,
        duration_s=frames / fps if fps else 0.0, codec=codec, rotation=rotation, pix_fmt=pix_fmt,
        color_trc=None, color_primaries=None, hdr=False, backend="opencv",
    )


def probe_video(path: str | Path) -> VideoInfo:
    """Video metadata via PyAV (OpenCV CAP_PROP values as fallback). width/height are in display
    orientation; rotation is FFmpeg's display-matrix angle in counter-clockwise degrees (-90 for an
    iPhone portrait clip). Raises FileNotFoundError, or ValueError if neither backend can read it."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"video not found: {p}")
    try:
        return _probe_pyav(p)
    except Exception as exc:  # PyAV missing or cannot parse the container
        log.warning("PyAV probe failed for %s (%s); trying OpenCV", p, exc)
        try:
            return _probe_opencv(p)
        except Exception as exc2:
            raise ValueError(f"cannot read video {p}: pyav: {exc}; opencv: {exc2}") from exc2


# --------------------------------------------------------------------------- HDR -> SDR


def _hlg_inverse_oetf(e: np.ndarray) -> np.ndarray:
    """HLG signal E' in [0, 1] -> normalised scene-linear light E in [0, 1] (ITU-R BT.2100)."""
    low = e * e / 3.0
    high = (np.exp((np.maximum(e, 0.5) - _HLG_C) / _HLG_A) + _HLG_B) / 12.0
    return np.where(e <= 0.5, low, high)


def _pq_eotf(e: np.ndarray) -> np.ndarray:
    """PQ signal E' in [0, 1] -> display light as a fraction of 10000 cd/m2 (SMPTE ST 2084)."""
    p = np.power(e, 1.0 / _PQ_M2)
    return np.power(np.maximum(p - _PQ_C1, 0.0) / (_PQ_C2 - _PQ_C3 * p), 1.0 / _PQ_M1)


_TRANSFERS = {"hlg": _hlg_inverse_oetf, "pq": _pq_eotf}

# Display light (fraction of peak) of a 75 % HLG grey after the OOTF: 0.2032, i.e. ~203 cd/m2 on a
# 1000 cd/m2 display. Computed here from the BT.2100 formulas, not copied.
_HLG_REF_WHITE = float(_hlg_inverse_oetf(np.float64(0.75)) ** _HLG_GAMMA)


@cache
def _u16_lut(kind: str) -> np.ndarray:
    """Linear light for every 16-bit code value (decoders hand us rgb48: a lookup beats exp/pow)."""
    return _TRANSFERS[kind](np.arange(65536, dtype=np.float64) / 65535.0).astype(np.float32)


@cache
def _srgb_lut() -> np.ndarray:
    """uint8 sRGB OETF of linear light quantised to 16 bits (error < 0.1 code value)."""
    x = np.arange(65536, dtype=np.float64) / 65535.0
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def _linear(rgb: np.ndarray, kind: str) -> np.ndarray:
    """uint16 (0..65535), uint8 (0..255) or float (0..1) HxWx3 signal -> float32 linear light."""
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB array, got {getattr(rgb, 'shape', type(rgb))}")
    if rgb.dtype == np.uint16:
        return _u16_lut(kind)[rgb]
    if rgb.dtype == np.uint8:
        x = rgb.astype(np.float32) / 255.0
    elif np.issubdtype(rgb.dtype, np.floating):
        x = np.clip(np.nan_to_num(rgb.astype(np.float32), nan=0.0), 0.0, 1.0)
    else:
        raise TypeError(f"unsupported dtype {rgb.dtype}")
    return _TRANSFERS[kind](x).astype(np.float32, copy=False)


def _bt2020_linear_to_sdr_bgr(lin: np.ndarray) -> np.ndarray:
    """Linear BT.2020 RGB with 1.0 = SDR reference white -> uint8 sRGB/BT.709 BGR."""
    rgb = lin @ _BT2020_TO_BT709.T
    np.maximum(rgb, 0.0, out=rgb)  # out-of-gamut negatives -> 0
    # Reinhard-style soft clip on max(R, G, B) above the knee: hue is kept, output stays < 1
    m = np.maximum(np.maximum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])
    x = np.maximum(m - SOFT_CLIP_KNEE, 0.0) / (1.0 - SOFT_CLIP_KNEE)
    compressed = SOFT_CLIP_KNEE + (1.0 - SOFT_CLIP_KNEE) * x / (1.0 + x)
    rgb *= np.where(m > SOFT_CLIP_KNEE, compressed / np.maximum(m, 1e-12), 1.0)[:, :, None]
    idx = (np.clip(rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)  # sRGB OETF via the lookup table
    return np.ascontiguousarray(_srgb_lut()[idx][:, :, ::-1])


def hlg_to_sdr(rgb: np.ndarray) -> np.ndarray:
    """HLG/BT.2020 RGB (uint16 0..65535 or float 0..1) -> SDR sRGB/BT.709 uint8 BGR.
    Steps: HLG inverse OETF -> OOTF (gamma 1.2) -> BT.2020->BT.709 3x3 -> scale so 75% HLG (~203 nit)
    maps to reference white -> soft clip (Reinhard) -> sRGB OETF -> uint8, RGB->BGR."""
    e = _linear(rgb, "hlg")  # scene light, BT.2020
    # OOTF with alpha = 1 (fraction of peak), then 75 % HLG grey -> 1.0
    gain = np.power(e @ _BT2020_LUMA, _HLG_GAMMA - 1.0) / _HLG_REF_WHITE
    return _bt2020_linear_to_sdr_bgr(e * gain[:, :, None])


def pq_to_sdr(rgb: np.ndarray) -> np.ndarray:
    """PQ (SMPTE ST 2084)/BT.2020 RGB (uint16 0..65535 or float 0..1) -> SDR sRGB/BT.709 uint8 BGR.
    Steps: PQ EOTF (cd/m2) -> divide by 203 cd/m2 reference white -> BT.2020->BT.709 -> soft clip -> sRGB."""
    return _bt2020_linear_to_sdr_bgr(_linear(rgb, "pq") * (10000.0 / PQ_REFERENCE_WHITE_NITS))


# --------------------------------------------------------------------------- extraction


def _sample_indices(frame_count: int, fps: float, target_fps: float, max_frames: int | None) -> list[int]:
    """Source frame indices nearest to t = k / target_fps (k = 0, 1, ...): a step of fps / target_fps
    frames (at least 1). If the grid has more than max_frames points, an evenly spaced subset of it
    that still spans the whole clip is kept (a walkaround keeps all sides of the car). With a step >= 1
    the grid is strictly increasing, so the subset is computed from k directly: memory is O(max_frames)
    even if a container header claims billions of frames."""
    if frame_count <= 0:
        return []
    step = max(fps / target_fps, 1.0)
    n = int(math.floor((frame_count - 1) / step + 1e-9)) + 1
    ks: Any = range(n)
    if max_frames is not None and n > max_frames:
        ks = np.linspace(0, n - 1, max_frames).round().astype(np.int64).tolist()
    return [min(int(math.floor(k * step + 0.5)), frame_count - 1) for k in ks]


def _count_frames(p: Path) -> int:
    """Frame count for containers without one: PyAV packet count, else an OpenCV grab() loop."""
    try:
        import av

        with av.open(str(p)) as c:
            s = c.streams.video[0]
            return sum(1 for pkt in c.demux(s) if pkt.size)
    except Exception:
        cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
        n = 0
        while cap.isOpened() and cap.grab():
            n += 1
        cap.release()
        return n


def _as_bgr8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"decoder returned {img.dtype} {img.shape}, expected uint8 HxWx3")
    return img


def _frame(img: np.ndarray, i: int, info: VideoInfo, max_side: int | None) -> Frame:
    return Frame(index=i, t_s=i / info.fps, image=resize_max_side(_as_bgr8(img), max_side), source=info.path)


def _missing(planned: list[int], decoded_upto: int) -> list[str]:
    left = [k for k in planned if k >= decoded_upto]
    return [f"stream ended after {decoded_upto} frames; frames {left} not decoded"] if left else []


def _decode_opencv(
    p: Path, info: VideoInfo, indices: list[int], max_side: int | None, errors: list[str]
) -> list[Frame]:
    cap = cv2.VideoCapture(str(p), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise ValueError("OpenCV cannot open the file")
    frames: list[Frame] = []
    wanted, i = set(indices), 0
    last = indices[-1] if indices else -1
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # rotate from the container display matrix
        while i <= last and cap.grab():
            if i in wanted:
                try:
                    ok, img = cap.retrieve()
                    if not ok or img is None:
                        raise ValueError("retrieve() returned no image")
                    frames.append(_frame(img, i, info, max_side))
                except Exception as exc:
                    errors.append(f"frame {i}: {type(exc).__name__}: {exc}")
            i += 1
    finally:
        cap.release()
    errors.extend(_missing(indices, i))
    return frames


def _pyav_bgr(f: Any, info: VideoInfo, max_side: int | None = None) -> np.ndarray:
    """Decoded PyAV frame -> uint8 BGR SDR in display orientation. HDR frames are downscaled by
    swscale (AREA) to max_side before the tone mapping, which is the expensive step."""
    trc = int(f.color_trc) if _known(f.color_trc) else _TRC_IDS.get(info.color_trc or "", 2)
    # pass the frame's own range (1 = limited/MPEG, 2 = full/JPEG); unspecified -> swscale default (limited)
    kw: dict[str, Any] = {"src_color_range": int(f.color_range)} if f.color_range else {}
    if trc in (_TRC_IDS[HLG_TRC], _TRC_IDS[PQ_TRC]):
        if max_side and max(f.width, f.height) > max_side:
            s = max_side / max(f.width, f.height)
            kw.update(width=max(1, round(f.width * s)), height=max(1, round(f.height * s)), interpolation="AREA")
        rgb48 = f.to_ndarray(format="rgb48le", **kw)
        img = hlg_to_sdr(rgb48) if trc == _TRC_IDS[HLG_TRC] else pq_to_sdr(rgb48)
    else:
        img = f.to_ndarray(format="bgr24", **kw)
    rotation = _norm_deg(f.rotation) if f.rotation else info.rotation
    if rotation % 360:
        # rotation is counter-clockwise (FFmpeg display matrix) and so is np.rot90:
        # -90 (iPhone portrait) -> k=3, i.e. 90 deg clockwise, the same as OpenCV's auto-rotation
        img = np.rot90(img, k=round(rotation / 90) % 4)
    return np.ascontiguousarray(img)


def _decode_pyav_pass(
    p: Path, info: VideoInfo, indices: list[int], max_side: int | None, errors: list[str], *, threaded: bool
) -> tuple[list[Frame], bool]:
    """One sequential PyAV decode. Returns (frames, clean); clean is False if a packet failed to demux or
    decode, or the stream ended before the last planned index (PyAV swallows a decode error during the final
    flush once it has received a frame in that call, and the frames still queued are lost silently).
    Single-threaded, a packet that fails to decode still counts as one source frame (one frame per packet in
    MP4/MOV), so later indices stay aligned with the source."""
    import av

    frames: list[Frame] = []
    wanted, i, clean = set(indices), 0, True
    last = indices[-1] if indices else -1
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO" if threaded else "SLICE"  # slice threads never hold frames back
        packets = c.demux(s)
        while i <= last:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except av.FFmpegError as exc:  # unreadable container data: keep the frames decoded so far
                errors.append(f"demux stopped after {i} frames: {type(exc).__name__}: {exc}")
                clean = False
                break
            try:
                decoded = packet.decode()
            except av.FFmpegError as exc:  # corrupt packet: skip it, keep going
                errors.append(f"packet pts={packet.pts}: {type(exc).__name__}: {exc}")
                clean = False
                if not threaded and packet.size:
                    i += 1  # its frame is lost; with frame threading the error surfaces later, so no count
                continue
            for f in decoded:
                if i in wanted:
                    try:
                        frames.append(_frame(_pyav_bgr(f, info, max_side), i, info, max_side))
                    except Exception as exc:
                        errors.append(f"frame {i}: {type(exc).__name__}: {exc}")
                i += 1
    errors.extend(_missing(indices, i))
    return frames, clean and i > last


def _decode_pyav(
    p: Path, info: VideoInfo, indices: list[int], max_side: int | None, errors: list[str]
) -> list[Frame]:
    """PyAV decode with frame threading. After a decode error, a frame-threaded decoder can lose every frame
    still queued in it: the error comes back during the final flush and the rest are never received (on a
    16-thread CPU, a corrupt packet near the end of a test clip cost the 13 decodable frames queued after it,
    with or without an exception). A pass with errors or a short read is therefore repeated
    single-threaded, which skips only the corrupt packets; the pass with more frames wins."""
    attempt: list[str] = []
    frames, clean = _decode_pyav_pass(p, info, indices, max_side, attempt, threaded=True)
    if not clean:
        retry: list[str] = []
        try:
            again, _ = _decode_pyav_pass(p, info, indices, max_side, retry, threaded=False)
        except Exception as exc:  # keep the threaded result
            attempt.append(f"single-threaded retry failed: {type(exc).__name__}: {exc}")
        else:
            if len(again) >= len(frames):  # equal counts: the single-threaded indices are aligned
                frames, attempt = again, ["frame-threaded decode incomplete; decoded again single-threaded", *retry]
    errors.extend(attempt)
    return frames


def extract_frames(
    path: str | Path,
    *,
    target_fps: float = 2.5,
    max_side: int | None = None,
    max_frames: int | None = None,
    backend: str = "auto",          # "auto" | "opencv" | "pyav"
) -> tuple[list[Frame], VideoInfo, list[str]]:
    """Sample frames at ~target_fps (evenly by timestamp). Returns (frames, info, errors).
    backend="auto": OpenCV for SDR, PyAV for HDR or if OpenCV fails to open the file.

    Frame index k*fps/target_fps (rounded) is taken for k = 0, 1, ...; max_frames keeps an evenly
    spaced subset of those that spans the whole clip; max_side downscales with INTER_AREA (HDR frames:
    swscale AREA before tone mapping). OpenCV stops reading at the first corrupt frame, so in auto mode
    an OpenCV read that comes up short (fewer frames than planned) hands over to PyAV, which skips corrupt
    packets, and the more complete result is returned; for HDR, OpenCV is only used if PyAV decodes no
    frame. A frame that fails to decode is skipped and described in ``errors``; a file that cannot be
    opened at all raises FileNotFoundError / ValueError. ``info.backend`` is the backend whose frames are
    returned; ``errors`` lists what happened in every attempt.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if not target_fps > 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")
    p = Path(path)
    info = probe_video(p)
    if info.fps <= 0:
        raise ValueError(f"cannot determine the frame rate of {p}")
    indices = _sample_indices(info.frame_count or _count_frames(p), info.fps, target_fps, max_frames)

    if backend == "auto":
        order = ["pyav", "opencv"] if info.hdr else ["opencv", "pyav"]
    else:
        order = [backend]
    errors: list[str] = []
    results: list[tuple[list[Frame], str]] = []
    for name in order:
        attempt: list[str] = []
        decode = _decode_opencv if name == "opencv" else _decode_pyav
        try:
            frames = decode(p, info, indices, max_side, attempt)
        except Exception as exc:  # this backend cannot open the file
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        errors.extend(attempt)
        if name == "opencv" and info.hdr and frames:
            errors.append(f"opencv: {info.color_trc} HDR video decoded without tone mapping")
        results.append((frames, name))
        if len(frames) >= len(indices) or (frames and name == "pyav") or name == order[-1]:
            break
        errors.append(f"{name}: {len(frames)} of {len(indices)} planned frames decoded, trying the next backend")
    if not results:
        return [], replace(info, backend=order[-1]), errors
    frames, name = max(results, key=lambda r: len(r[0]))  # the first of equally complete results
    return frames, replace(info, backend=name), errors
