from __future__ import annotations

import hashlib
import os
import stat
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

OFFICIAL_REPOSITORY = "https://github.com/xpwu95/IP102"
OFFICIAL_SOURCES = {
    "google_drive": (
        "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo?usp=sharing"
    ),
    "aliyun_drive": "https://www.aliyundrive.com/s/c5G9scSGyak",
}
ACADEMIC_USE_NOTICE = (
    "IP102 is free for academic use. The authors request contact for other purposes. "
    f"Review the current terms at {OFFICIAL_REPOSITORY} before acquiring the data."
)


class AcquisitionError(RuntimeError):
    """Raised when an archive cannot be acquired or extracted safely."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    progress: Callable[[int], None] | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "PestScope-IP102/0.1"})

    try:
        downloaded = 0
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            for chunk in iter(lambda: response.read(chunk_size), b""):
                output.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded)
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(f"Download failed for {url}: {exc}") from exc

    return destination


def verify_sha256(path: Path, expected: str | None) -> str:
    actual = sha256_file(path)
    if expected and actual.lower() != expected.strip().lower():
        raise AcquisitionError(
            f"SHA-256 mismatch for {path.name}: expected {expected.lower()}, got {actual}"
        )
    return actual


def _safe_target(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    if normalized.startswith("/"):
        raise AcquisitionError(f"Archive contains an absolute path: {member_name}")

    target = (root / normalized).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise AcquisitionError(f"Archive path escapes destination: {member_name}") from exc
    return target


def _extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise AcquisitionError(f"Archive contains a symbolic link: {member.filename}")
            _safe_target(destination, member.filename)
        bundle.extractall(destination)


def _extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, mode="r:*") as bundle:
        members = bundle.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise AcquisitionError(f"Archive contains a link: {member.name}")
            _safe_target(destination, member.name)
        bundle.extractall(destination, members=members)


def extract_archive(archive: Path, destination: Path) -> Path:
    archive = archive.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        _extract_zip(archive, destination)
    elif tarfile.is_tarfile(archive):
        _extract_tar(archive, destination)
    else:
        raise AcquisitionError(f"Unsupported archive {archive.name}. Use ZIP, TAR, TAR.GZ, or TGZ.")
    return destination


def acquire_archive(
    *,
    destination: Path,
    accept_academic_use: bool,
    archive: Path | None = None,
    url: str | None = None,
    expected_sha256: str | None = None,
    keep_archive: bool = True,
) -> dict[str, str]:
    if not accept_academic_use:
        raise AcquisitionError(
            "Academic-use acknowledgement is required. Review the official terms and pass "
            "--accept-academic-use."
        )
    if (archive is None) == (url is None):
        raise AcquisitionError("Provide exactly one of --archive or --url")

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = False

    if url:
        filename = Path(urlparse(url).path).name or "ip102-download.zip"
        archive_path = download_file(url, destination.parent / filename)
        downloaded = True
    else:
        archive_path = archive.resolve()
        if not archive_path.is_file():
            raise AcquisitionError(f"Archive does not exist: {archive_path}")

    digest = verify_sha256(archive_path, expected_sha256)
    extract_archive(archive_path, destination)

    if downloaded and not keep_archive:
        archive_path.unlink(missing_ok=True)

    return {
        "archive": str(archive_path),
        "destination": str(destination),
        "sha256": digest,
        "source": url or str(archive_path),
    }
