from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml
from PIL import Image, ImageOps, UnidentifiedImageError

from pestscope.inference.service import InferenceService


@dataclass(frozen=True)
class ExternalRecord:
    class_id: int
    canonical_name: str
    common_name_vi: str
    provider: str
    license: str
    source_url: str
    status: str
    image_path: str | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "canonical_name": self.canonical_name,
            "common_name_vi": self.common_name_vi,
            "provider": self.provider,
            "license": self.license,
            "source_url": self.source_url,
            "status": self.status,
            "image_path": self.image_path,
            "error": self.error,
        }


def _download(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "PestScope-IP102-external-benchmark/0.1 "
                "(student research; contact via repository owner)"
            )
        },
    )
    with urlopen(request, timeout=15) as response:
        return response.read(12 * 1024 * 1024)


def _valid_cached_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        return False
    return True


def _looks_like_wikimedia(url: str) -> bool:
    return urlparse(url).netloc.endswith("wikimedia.org")


def _normalize_image(content: bytes, output: Path) -> tuple[str, str | None]:
    try:
        with Image.open(BytesIO(content)) as image:
            clean = ImageOps.exif_transpose(image).convert("RGB")
            clean.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            output.parent.mkdir(parents=True, exist_ok=True)
            clean.save(output, format="JPEG", quality=88, optimize=True)
        return "ok", None
    except (UnidentifiedImageError, OSError) as exc:
        return "skipped", f"unsupported_or_invalid_image: {exc}"


def build_external_benchmark(
    *,
    class_review_path: Path,
    output_dir: Path,
) -> dict:
    raw = yaml.safe_load(class_review_path.read_text(encoding="utf-8"))
    image_dir = output_dir / "images"
    records = []
    previous_wikimedia_download = False
    for item in raw["classes"]:
        source = item["external_source"]
        class_id = int(item["ip102_id"])
        output = image_dir / f"class-{class_id}.jpg"
        url = str(source["url"])
        if _valid_cached_image(output):
            status, error = "ok", None
        else:
            if previous_wikimedia_download and _looks_like_wikimedia(url):
                # Wikimedia may reject bursty direct-file requests; keep the builder polite.
                import time

                time.sleep(1.0)
            try:
                status, error = _normalize_image(_download(url), output)
            except (OSError, URLError, TimeoutError) as exc:
                status, error = "failed", str(exc)
            previous_wikimedia_download = _looks_like_wikimedia(url)
        records.append(
            ExternalRecord(
                class_id=class_id,
                canonical_name=str(item["canonical_name"]),
                common_name_vi=str(item["common_name_vi"]),
                provider=str(source["provider"]),
                license=str(source["license"]),
                source_url=str(source["url"]),
                status=status,
                image_path=str(output) if status == "ok" else None,
                error=error,
            ).to_dict()
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(class_review_path),
        "records": records,
        "ok_count": sum(1 for record in records if record["status"] == "ok"),
        "skipped_count": sum(1 for record in records if record["status"] != "ok"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def evaluate_external_benchmark(
    *,
    bundle_dir: Path,
    manifest_path: Path,
    device: str = "cpu",
    accept_threshold: float | None = None,
    uncertain_threshold: float | None = None,
) -> dict:
    service = InferenceService.from_bundle(
        bundle_dir,
        device=device,
        accept_threshold=accept_threshold,
        uncertain_threshold=uncertain_threshold,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for record in manifest["records"]:
        if record["status"] != "ok":
            rows.append({**record, "evaluated": False})
            continue
        with Image.open(record["image_path"]) as image:
            prediction = service.predict(image.convert("RGB"), top_k=3)
        top_k = prediction["top_k"]
        top_ids = [item["class_id"] for item in top_k]
        rows.append(
            {
                **record,
                "evaluated": True,
                "decision": prediction["decision"],
                "confidence": prediction["confidence"],
                "predicted_class_id": top_ids[0],
                "top3_class_ids": top_ids,
                "top1_correct": top_ids[0] == record["class_id"],
                "top3_correct": record["class_id"] in top_ids,
            }
        )

    evaluated = [row for row in rows if row.get("evaluated")]
    accepted = [row for row in evaluated if row["decision"] == "accepted"]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "model_version": service.metadata.get("run_id"),
        "records": rows,
        "summary": {
            "evaluated": len(evaluated),
            "top1_accuracy": (
                sum(1 for row in evaluated if row["top1_correct"]) / len(evaluated)
                if evaluated
                else 0.0
            ),
            "top3_accuracy": (
                sum(1 for row in evaluated if row["top3_correct"]) / len(evaluated)
                if evaluated
                else 0.0
            ),
            "accepted_count": len(accepted),
            "accepted_precision": (
                sum(1 for row in accepted if row["top1_correct"]) / len(accepted)
                if accepted
                else 0.0
            ),
            "unsupported_rate": (
                sum(1 for row in evaluated if row["decision"] == "unsupported") / len(evaluated)
                if evaluated
                else 0.0
            ),
        },
    }
    output = manifest_path.with_name("evaluation.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
