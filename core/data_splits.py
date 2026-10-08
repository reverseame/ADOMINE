"""Single source of truth for the D1 / D2 / D3 dataset partitioning.

- D1: training set for the adversarial DGA models.
- D2: training set for the detectors.
- D3: control groups used as evaluation baselines (the test set).

The three roles are disjoint on both the benign side (Tranco) and the malicious
side (DGArchive). The partition is materialised once by
`tools/build_datasets.py` into `dataset/splits/`, which this module reads. No
model or detector reads the raw Tranco or DGArchive files.

Each split holds the same number of benign and of malicious domains. D1 and D2
draw their malicious half equally from every family file; D3 instead holds one
file per control family (Qakbot, Rovnix, Conficker and Cryptolocker arithmetic;
Suppobox and Gozi dictionary; Dyre and Bamital hash; Banjori permutation; Symmi
pronounceable), so evasion can be reported per family. The manifest lists them
under `d3_families`.

To rebuild the datasets (different seed or sizes), run
`python3 tools/build_datasets.py --force` and retrain the detectors and the
adversarial models: their cached weights belong to the previous partition.
"""

import hashlib
import json
import logging
import os
import random
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
SPLITS_DIR = os.path.join(PROJECT_ROOT, "dataset", "splits")
MANIFEST_PATH = os.path.join(SPLITS_DIR, "manifest.json")

# Split identifiers, kept as named constants so call sites name the split they read.
TRANCO_D1_BENIGN = "D1"
TRANCO_D2_BENIGN = "D2"
TRANCO_D3_BENIGN = "D3"

DGARCHIVE_D1_MALICIOUS = "D1"
DGARCHIVE_D2_MALICIOUS = "D2"
DGARCHIVE_D3_MALICIOUS = "D3"

# Detector (D2) training draw. Every detector uses the defaults so they train on
# the identical, order-independent sample.
D2_MALICIOUS_TRAINING_TOTAL = 100_000
D2_MALICIOUS_TRAINING_SEED = 42

# Adversarial-model seed draw (D1 malicious).
MALICIOUS_SEEDS_SEED = 42


def _read_domains(path: str) -> List[str]:
    """Read one domain per line, ignoring blanks. The domain is the last
    comma-separated field, so both plain and `date,domain` layouts work."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found at {path}. Build the splits first: "
            "python3 tools/build_datasets.py"
        )
    domains: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip().rsplit(",", 1)[-1].strip()
            if value:
                domains.append(value)
    return domains


def _sample(domains: List[str], total: Optional[int], seed: int, label: str) -> List[str]:
    """Return `total` domains drawn with a private RNG, or all of them when
    `total` is None or covers the whole set. The RNG is local, so the result
    does not depend on global RNG state or on call order."""
    if total is None or total >= len(domains):
        if total is not None and total > len(domains):
            logger.warning(
                "%s asked for %d domains but the split holds %d; returning all of them.",
                label, total, len(domains),
            )
        return list(domains)
    return random.Random(seed).sample(domains, total)


def load_tranco(split: str) -> List[str]:
    """Return the benign domains of one split (`D1`, `D2` or `D3`)."""
    return _read_domains(os.path.join(SPLITS_DIR, f"{split}_benign.csv"))


def load_dgarchive(family: str, split: str = DGARCHIVE_D3_MALICIOUS) -> List[str]:
    """Return the malicious domains of one control family in a split.

    `family` is the DGArchive file name (for example `dyre_dga.csv`). Only D3
    is stored per family; D1 and D2 mix every family in a single file and are
    read with load_malicious_seeds and load_d2_malicious_training.
    """
    return _read_domains(os.path.join(SPLITS_DIR, f"{split}_malicious_{family}"))


def load_d2_malicious_training(
    total: Optional[int] = D2_MALICIOUS_TRAINING_TOTAL,
    seed: int = D2_MALICIOUS_TRAINING_SEED,
) -> List[str]:
    """Return the detectors' malicious training half (the D2 malicious split).

    The split already holds an equal-per-family draw over the 58 family files.
    Every detector calls this with the defaults, so they train on the identical
    sample regardless of call order.
    """
    domains = _read_domains(os.path.join(SPLITS_DIR, "D2_malicious.csv"))
    return _sample(domains, total, seed, "D2 malicious training draw")


def load_malicious_seeds(
    total: Optional[int] = None,
    seed: int = MALICIOUS_SEEDS_SEED,
) -> List[str]:
    """Return real AGDs (full domains, TLD kept) from the D1 malicious split.

    Used by the adversarial models that mutate or seed their generation from
    known malicious domains (MaskDGA, FGSM-based DGA, GADGA, DnGAN, CLETer,
    GeometricPerturbationDGA, WGAN-based DGA, WordDGA). `total` takes a
    deterministic subsample; omit it to get the whole split.
    """
    domains = _read_domains(os.path.join(SPLITS_DIR, "D1_malicious.csv"))
    return _sample(domains, total, seed, "malicious seed draw")


def load_manifest() -> Dict:
    """Return the manifest written by tools/build_datasets.py (seed, sources,
    per-file domain counts and sha256)."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_on_import() -> None:
    """Check that the materialised splits are present and unmodified.

    Only the manifest is trusted for sizes and hashes, so this costs a few
    tens of milliseconds.
    """
    if not os.path.isdir(SPLITS_DIR) or not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"No dataset splits under {SPLITS_DIR}. Build them first: "
            "python3 tools/build_datasets.py"
        )
    manifest = load_manifest()
    for name, meta in manifest["files"].items():
        path = os.path.join(SPLITS_DIR, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file listed in the manifest is missing: {path}")
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != meta["sha256"]:
            logger.warning(
                "%s does not match the manifest hash; the splits were modified after "
                "the build, so results are not traceable to the recorded datasets.", name,
            )


_validate_on_import()
