import re
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Union

ProgressFn = Callable[[str, float], None]
from urllib.parse import urlparse

import requests

from class_assessor.config import MAX_VIDEO_BYTES

DRIVE_FILE_RE = re.compile(
    r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
)
DRIVE_OPEN_RE = re.compile(
    r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
)
DRIVE_UC_RE = re.compile(
    r"https?://drive\.google\.com/uc\?(?:export=download&)?id=([a-zA-Z0-9_-]+)",
)


def _suffix_from_url(url: str) -> str:
    path = urlparse(url).path
    for ext in (".mp4", ".webm", ".mkv", ".mov"):
        if path.lower().endswith(ext):
            return ext
    return ".mp4"


def _download_http(
    url: str, dest: Path, progress: Optional[ProgressFn] = None
) -> None:
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = 0
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise ValueError(
                        f"Download exceeds limit ({MAX_VIDEO_BYTES // (1024*1024)} MB)"
                    )
                f.write(chunk)
                if progress:
                    progress(
                        f"Downloading… {total // (1024 * 1024)} MB received",
                        0.02 + min(0.06, total / MAX_VIDEO_BYTES * 0.06),
                    )


def _extract_drive_file_id(url: str) -> Optional[str]:
    for pattern in (DRIVE_FILE_RE, DRIVE_OPEN_RE, DRIVE_UC_RE):
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def _download_google_drive(
    url: str, dest: Path, progress: Optional[ProgressFn] = None
) -> None:
    import gdown

    if progress:
        progress("Downloading from Google Drive…", 0.03)
    file_id = _extract_drive_file_id(url)
    if not file_id:
        raise ValueError(
            "Use a direct Google Drive **file** link "
            "(https://drive.google.com/file/d/FILE_ID/view…), not a folder link."
        )
    gdown.download(id=file_id, output=str(dest), quiet=False)
    if not dest.exists() or dest.stat().st_size == 0:
        raise ValueError("Google Drive download failed or file is empty.")
    if dest.stat().st_size > MAX_VIDEO_BYTES:
        dest.unlink(missing_ok=True)
        raise ValueError("Download exceeds size limit")


def materialize_video(
    source: Union[str, BinaryIO, bytes],
    *,
    suffix: str = ".mp4",
    progress: Optional[ProgressFn] = None,
) -> Path:
    """
    Return path to a temp video file. Caller should delete parent temp dir when done.
    """
    tmp = Path(tempfile.mkdtemp(prefix="class_assess_"))
    dest = tmp / f"recording{suffix}"

    if isinstance(source, str):
        url = source.strip()
        if url.startswith("http://") or url.startswith("https://"):
            if "drive.google.com" in url or "docs.google.com" in url:
                _download_google_drive(url, dest, progress)
            else:
                dest = tmp / f"recording{_suffix_from_url(url)}"
                _download_http(url, dest, progress)
        else:
            p = Path(url)
            if not p.is_file():
                raise FileNotFoundError(f"Not a file: {url}")
            shutil.copy2(p, dest)
    else:
        if progress:
            progress("Saving uploaded video to temp folder…", 0.04)
        if hasattr(source, "read"):
            data = source.read()
        else:
            data = source
        dest.write_bytes(data)

    if not dest.exists() or dest.stat().st_size == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError("Empty or missing video file")
    return dest
