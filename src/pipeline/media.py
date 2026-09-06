from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def probe_video(path: Path) -> tuple[float, float]:
    """Return `(fps, duration_s)`."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"Video not found: {path}")
    for probe in (_probe_cv2, _probe_av, _probe_ffprobe):
        result = probe(path)
        if result is not None:
            fps, duration = result
            if fps > 0 and duration > 0:
                return float(fps), float(duration)
    raise MediaError(f"Could not read video metadata: {path}")


def extract_clip_mp4(
    source: Path,
    dest: Path,
    start_sec: float,
    end_sec: float,
) -> Path | None:
    """Cut `[start_sec, end_sec]` into `dest`. Returns dest or None if cut failed."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if end_sec <= start_sec:
        return None
    if _extract_ffmpeg(source, dest, start_sec, end_sec):
        return dest
    if _extract_av(source, dest, start_sec, end_sec):
        return dest
    return None


def _probe_cv2(path: Path) -> tuple[float, float] | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0 or n <= 0:
        return None
    return fps, n / fps


def _probe_av(path: Path) -> tuple[float, float] | None:
    try:
        import av
    except ImportError:
        return None
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0.0)
        duration = float(container.duration or 0) / 1_000_000 if container.duration else 0.0
        if duration <= 0 and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        container.close()
    except Exception:
        return None
    if fps <= 0 or duration <= 0:
        return None
    return fps, duration


def _probe_ffprobe(path: Path) -> tuple[float, float] | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        dur_raw = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            text=True,
        ).strip()
        fps_raw = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "csv=p=0",
                str(path),
            ],
            text=True,
        ).strip()
        duration = float(dur_raw)
        if "/" in fps_raw:
            num, den = fps_raw.split("/", 1)
            fps = float(num) / float(den) if float(den) else 0.0
        else:
            fps = float(fps_raw)
    except (subprocess.CalledProcessError, ValueError):
        return None
    return fps, duration


def _extract_ffmpeg(source: Path, dest: Path, start_sec: float, end_sec: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return dest.is_file() and dest.stat().st_size > 0


def _extract_av(source: Path, dest: Path, start_sec: float, end_sec: float) -> bool:
    try:
        import av
    except ImportError:
        return False
    try:
        inp = av.open(str(source))
        in_stream = inp.streams.video[0]
        in_stream.thread_type = "AUTO"
        out = av.open(str(dest), mode="w")
        out_stream = out.add_stream("libx264", rate=in_stream.average_rate or 15)
        out_stream.width = in_stream.width
        out_stream.height = in_stream.height
        out_stream.pix_fmt = "yuv420p"
        for frame in inp.decode(in_stream):
            ts = float(frame.time or 0.0)
            if ts < start_sec:
                continue
            if ts > end_sec:
                break
            for packet in out_stream.encode(frame):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)
        out.close()
        inp.close()
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False
    return dest.is_file() and dest.stat().st_size > 0
