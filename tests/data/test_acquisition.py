from __future__ import annotations

import zipfile

import pytest

from pestscope.data.acquisition import (
    AcquisitionError,
    acquire_archive,
    extract_archive,
    sha256_file,
)


def test_acquire_local_archive_requires_acknowledgement_and_verifies_hash(tmp_path) -> None:
    archive = tmp_path / "ip102.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("classification/classes.txt", "1 rice leaf roller\n")
    digest = sha256_file(archive)

    with pytest.raises(AcquisitionError, match="acknowledgement"):
        acquire_archive(
            destination=tmp_path / "rejected",
            archive=archive,
            accept_academic_use=False,
        )

    result = acquire_archive(
        destination=tmp_path / "accepted",
        archive=archive,
        accept_academic_use=True,
        expected_sha256=digest,
    )

    assert result["sha256"] == digest
    assert (tmp_path / "accepted/classification/classes.txt").is_file()


def test_extract_archive_rejects_path_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside.txt", "unsafe")

    with pytest.raises(AcquisitionError, match="escapes destination"):
        extract_archive(archive, tmp_path / "output")
