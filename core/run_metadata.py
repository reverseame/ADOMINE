"""Run metadata stamped into every analysis JSON for reproducibility."""

import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict

from .data_splits import MANIFEST_PATH, load_manifest


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_metadata() -> Dict[str, Any]:
    """Identify the D1/D2/D3 splits the run consumed.

    The manifest lists the sha256 of every split file, so hashing the manifest
    itself pins the whole partition in one value; the build seed and timestamp
    make it reproducible with tools/build_datasets.py.
    """
    manifest = load_manifest()
    return {
        "dataset_manifest_sha256": _sha256_file(MANIFEST_PATH),
        "dataset_build_seed": manifest["seed"],
        "dataset_built_utc": manifest["created_utc"],
        "dataset_benign_source": manifest["benign_source"],
        "dataset_malicious_source": manifest["malicious_source"],
    }


def collect_run_metadata(seed: int) -> Dict[str, Any]:
    """Collect once per run: timestamp, seed, tool versions, dataset hashes."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python_version": platform.python_version(),
        "tensorflow_version": _package_version("tensorflow"),
        "numpy_version": _package_version("numpy"),
        "tldextract_version": _package_version("tldextract"),
        **_dataset_metadata(),
    }
