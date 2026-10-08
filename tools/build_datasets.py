#!/usr/bin/env python3
"""Build the three disjoint evaluation datasets (D1, D2, D3).

- D1: training set for the adversarial DGA models.
- D2: training set for the detectors.
- D3: control groups used as evaluation baselines (the test set).

Each dataset holds the same number of benign domains (Tranco) and of malicious
domains (DGArchive). D1 and D2 draw their malicious half equally from every
family file, so models and detectors see the full diversity of DGAs. D3 instead
uses control families that each represent one generation scheme (Qakbot and
Rovnix arithmetic, Suppobox dictionary and Dyre hash in a full build; further
families such as Conficker, Cryptolocker, Banjori, Symmi, Gozi and Bamital are
appended with --add-d3-families), with a full-size sample per family so evasion
can be reported per family.

Sampling is random and reproducible: one RNG per source, seeded from --seed, so
the output does not depend on file order. Every domain is used at most once
across the three datasets, and the script verifies that before writing.

Extra control families can be appended to an existing build without touching
any file already written (a rebuild would redraw D1 and D2 for those families,
since one reservoir per family feeds the three datasets). The incremental mode
draws each new family with its own RNG, excludes every domain already in the
splits and only adds D3_malicious_<family>.csv entries to the manifest.

Usage:
    python3 tools/build_datasets.py                 # defaults below
    python3 tools/build_datasets.py --seed 7 --benign-per-dataset 10000
    python3 tools/build_datasets.py --add-d3-families conficker_dga.csv gozi_dga.csv
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Set

logger = logging.getLogger("build_datasets")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
TRANCO_CSV = os.path.join(PROJECT_ROOT, "dataset", "top-1m.csv")
DGARCHIVE_DIR = os.path.join(PROJECT_ROOT, "dataset", "dgarchive")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "splits")

# Four families covering the distinct generation schemes, used as the D3
# malicious control groups (one model per family in main.py).
D3_FAMILIES = ("qakbot_dga.csv", "rovnix_dga.csv", "suppobox_dga.csv", "dyre_dga.csv")

# Oversampling factor for the reservoir, so that duplicates removed afterwards
# still leave enough unique domains.
OVERSAMPLE = 1.3


def reservoir_sample(path: str, k: int, rng: random.Random) -> List[str]:
    """Return k domains drawn uniformly at random from every line of path.

    Single pass, O(k) memory: the file is never loaded whole, which matters for
    the largest family files (tens of millions of rows). The domain is the last
    comma-separated field, matching both the 'domain' and 'date,domain' layouts.
    """
    reservoir: List[str] = []
    seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            domain = line.strip().rsplit(",", 1)[-1].strip().lower()
            if not domain:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(domain)
            else:
                j = rng.randrange(seen)
                if j < k:
                    reservoir[j] = domain
    return reservoir


def unique_preserving(domains: List[str]) -> List[str]:
    """Deduplicate while keeping the drawn order (a family can repeat a domain
    across dates)."""
    return list(dict.fromkeys(domains))


def split_quotas(total: int, parts: int) -> List[int]:
    """Split total into parts as evenly as possible (the remainder goes to the
    first buckets), so the datasets hold exactly `total` domains."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def load_benign(path: str, needed: int, rng: random.Random,
                exclude: Optional[Set[str]] = None) -> List[str]:
    """Return `needed` distinct Tranco domains drawn at random from the whole list.

    Drawing at random rather than by rank avoids a popularity bias: the top of
    the list holds shorter and more common names than the tail.

    The Tranco list is known to contain some malicious domains (Le Pochat et
    al., 2019), so any domain that this build already selected as malicious is
    excluded: a domain cannot carry both labels. Note this only removes the
    conflicts with the domains actually drawn here, not every Tranco entry that
    appears somewhere in DGArchive.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        domains = [line.strip().rsplit(",", 1)[-1].strip().lower() for line in fh]
    domains = unique_preserving([d for d in domains if d])
    if exclude:
        clean = [d for d in domains if d not in exclude]
        dropped = len(domains) - len(clean)
        if dropped:
            logger.info(f"[*] Dropped {dropped:,} Tranco domains also drawn as malicious.")
        domains = clean
    if len(domains) < needed:
        raise SystemExit(
            f"Tranco holds {len(domains)} usable distinct domains, {needed} required."
        )
    return rng.sample(domains, needed)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_domains(path: str, domains: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for domain in domains:
            fh.write(f"{domain}\n")


def read_domains(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def add_d3_families(out_dir: str, new_families: List[str], per_family: int) -> int:
    """Append control families to an existing build, leaving every written
    file byte-identical. Only the manifest changes (new entries and families)."""
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"No manifest under {out_dir}; build the datasets first.")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    seed = manifest["seed"]
    present = set(manifest["d3_families"])
    available = set(f for f in os.listdir(DGARCHIVE_DIR) if f.endswith(".csv"))
    for family in new_families:
        if family not in available:
            raise SystemExit(f"{family} is not under {DGARCHIVE_DIR}")
        if family in present:
            raise SystemExit(f"{family} is already a D3 control family")

    # Every domain already written, whichever dataset or label, is off limits.
    used: Set[str] = set()
    for name in manifest["files"]:
        used.update(read_domains(os.path.join(out_dir, name)))
    logger.info(f"[*] {len(used):,} domains already used by the existing splits")

    added: Dict[str, List[str]] = {}
    for family in new_families:
        # Own RNG stream, distinct from the one the full build uses for this family.
        rng = random.Random(f"{seed}:{family}:d3")
        drawn = unique_preserving(reservoir_sample(
            os.path.join(DGARCHIVE_DIR, family), int(per_family * OVERSAMPLE), rng))
        fresh = [d for d in drawn if d not in used][:per_family]
        if len(fresh) < per_family:
            logger.warning(f"[!] {family}: only {len(fresh):,} fresh domains of {per_family:,}")
        used.update(fresh)
        added[family] = fresh
        logger.info(f"  {family:<28} drawn={len(drawn):>7,} D3={len(fresh):,}")

    for family, domains in added.items():
        assert len(set(domains)) == len(domains), f"duplicates inside {family}"
        name = f"D3_malicious_{family}"
        path = os.path.join(out_dir, name)
        write_domains(path, domains)
        manifest["files"][name] = {"domains": len(domains), "sha256": sha256_of(path)}
        manifest["d3_families"].append(family)
        logger.info(f"  -> {name}: {len(domains):,} domains")
    manifest.setdefault("d3_families_added", []).append({
        "added_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": list(added),
    })
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)
    logger.info(f"\n[+] Added {len(added)} D3 families. Manifest: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42,
                        help="master seed; every draw derives its RNG from it (default: 42)")
    parser.add_argument("--benign-per-dataset", type=int, default=100_000,
                        help="benign domains in each of D1, D2, D3 (default: 100000)")
    parser.add_argument("--malicious-per-dataset", type=int, default=100_000,
                        help="malicious domains in each of D1 and D2, spread over every family (default: 100000)")
    parser.add_argument("--d3-per-family", type=int, default=100_000,
                        help="malicious domains per D3 control family (default: 100000)")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help="output directory (default: dataset/splits)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output directory")
    parser.add_argument("--add-d3-families", nargs="+", metavar="FAMILY_CSV",
                        help="append these DGArchive families as D3 control groups to an "
                             "existing build without rewriting any other file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    start = time.time()

    if args.add_d3_families:
        return add_d3_families(args.out, args.add_d3_families, args.d3_per_family)

    if os.path.isdir(args.out) and os.listdir(args.out) and not args.force:
        raise SystemExit(f"{args.out} already exists and is not empty. Use --force to rebuild.")
    os.makedirs(args.out, exist_ok=True)

    families = sorted(f for f in os.listdir(DGARCHIVE_DIR) if f.endswith(".csv"))
    if not families:
        raise SystemExit(f"No family CSVs under {DGARCHIVE_DIR}")
    missing = [f for f in D3_FAMILIES if f not in families]
    if missing:
        raise SystemExit(f"D3 families missing from {DGARCHIVE_DIR}: {missing}")

    logger.info(f"[*] {len(families)} DGArchive families, D3 control families: {list(D3_FAMILIES)}")

    # ------------------------------------------------------------- malicious
    # Drawn before the benign half so the benign draw can exclude any domain
    # already labelled malicious here.
    # Per-family quota for D1 and D2. The remainder of the integer division is
    # spread over the first families so each dataset holds exactly the target.
    quotas = split_quotas(args.malicious_per_dataset, len(families))
    per_family_quota = dict(zip(families, quotas))

    malicious: Dict[str, List[str]] = {"D1": [], "D2": []}
    d3_malicious: Dict[str, List[str]] = {}
    short_families: List[str] = []
    # DGArchive families are not disjoint: the same domain can appear in two
    # family files (related families such as murofet/murofetweekly or
    # nymaim/nymaim2). Uniqueness is therefore enforced globally, not per
    # family: a domain already taken is skipped. Families are visited in sorted
    # order, so which family keeps a shared domain is deterministic.
    used: Set[str] = set()
    cross_family_skipped = 0

    for idx, family in enumerate(families, 1):
        quota = per_family_quota[family]
        d3_quota = args.d3_per_family if family in D3_FAMILIES else 0
        needed = 2 * quota + d3_quota
        # One RNG per family: the draw does not depend on processing order.
        rng = random.Random(f"{args.seed}:{family}")
        drawn = unique_preserving(reservoir_sample(
            os.path.join(DGARCHIVE_DIR, family), int(needed * OVERSAMPLE), rng))
        fresh = [d for d in drawn if d not in used]
        cross_family_skipped += len(drawn) - len(fresh)

        take_d1 = fresh[:quota]
        take_d2 = fresh[quota:2 * quota]
        take_d3 = fresh[2 * quota:2 * quota + d3_quota]
        used.update(take_d1)
        used.update(take_d2)
        used.update(take_d3)

        malicious["D1"].extend(take_d1)
        malicious["D2"].extend(take_d2)
        if family in D3_FAMILIES:
            d3_malicious[family] = take_d3
        if len(take_d1) < quota or len(take_d2) < quota or len(take_d3) < d3_quota:
            short_families.append(
                f"{family}(D1 {len(take_d1)}/{quota}, D2 {len(take_d2)}/{quota}"
                + (f", D3 {len(take_d3)}/{d3_quota}" if d3_quota else "") + ")")
        logger.info(f"  [{idx:2d}/{len(families)}] {family:<28} drawn={len(drawn):>7,} "
                    f"D1={len(take_d1):,} D2={len(take_d2):,}"
                    + (f" D3={len(take_d3):,}" if family in D3_FAMILIES else ""))

    if cross_family_skipped:
        logger.info(f"[*] Skipped {cross_family_skipped:,} domains already taken by an earlier family.")
    if short_families:
        logger.warning(f"[!] Families that could not fill their quota: {short_families}")

    # ---------------------------------------------------------------- benign
    logger.info(f"[*] Drawing {3 * args.benign_per_dataset:,} distinct benign domains from Tranco...")
    benign_rng = random.Random(args.seed)
    benign = load_benign(TRANCO_CSV, 3 * args.benign_per_dataset, benign_rng, exclude=used)
    benign_sets = {
        "D1": benign[: args.benign_per_dataset],
        "D2": benign[args.benign_per_dataset: 2 * args.benign_per_dataset],
        "D3": benign[2 * args.benign_per_dataset:],
    }

    # ---------------------------------------------------------- verification
    logger.info("[*] Verifying the three datasets are disjoint...")
    all_malicious = {"D1": set(malicious["D1"]), "D2": set(malicious["D2"]),
                     "D3": set().union(*d3_malicious.values()) if d3_malicious else set()}
    for a, b in (("D1", "D2"), ("D1", "D3"), ("D2", "D3")):
        overlap_b = set(benign_sets[a]) & set(benign_sets[b])
        overlap_m = all_malicious[a] & all_malicious[b]
        assert not overlap_b, f"benign overlap {a}/{b}: {len(overlap_b)}"
        assert not overlap_m, f"malicious overlap {a}/{b}: {len(overlap_m)}"
    for name, domains in benign_sets.items():
        assert len(set(domains)) == len(domains), f"duplicates inside {name} benign"
    for name, domains in malicious.items():
        assert len(set(domains)) == len(domains), f"duplicates inside {name} malicious"
    d3_total = sum(len(v) for v in d3_malicious.values())
    assert len(all_malicious["D3"]) == d3_total, "the D3 control families share domains"
    for name in ("D1", "D2"):
        overlap = set(benign_sets[name]) & all_malicious[name]
        assert not overlap, f"a domain is both benign and malicious in {name}: {len(overlap)}"

    # --------------------------------------------------------------- writing
    manifest: Dict[str, object] = {
        "seed": args.seed,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benign_source": os.path.relpath(TRANCO_CSV, PROJECT_ROOT),
        "malicious_source": os.path.relpath(DGARCHIVE_DIR, PROJECT_ROOT),
        "families": len(families),
        "d3_families": list(D3_FAMILIES),
        "files": {},
    }

    def emit(name: str, domains: List[str]) -> None:
        path = os.path.join(args.out, name)
        write_domains(path, domains)
        manifest["files"][name] = {"domains": len(domains), "sha256": sha256_of(path)}
        logger.info(f"  -> {name}: {len(domains):,} domains")

    for name, domains in benign_sets.items():
        emit(f"{name}_benign.csv", domains)
    for name in ("D1", "D2"):
        emit(f"{name}_malicious.csv", malicious[name])
    for family, domains in d3_malicious.items():
        emit(f"D3_malicious_{family}", domains)

    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=4)

    logger.info(f"\n[+] Done in {time.time() - start:.0f}s. Manifest: {manifest_path}")
    logger.info(f"    D1: {len(benign_sets['D1']):,} benign + {len(malicious['D1']):,} malicious")
    logger.info(f"    D2: {len(benign_sets['D2']):,} benign + {len(malicious['D2']):,} malicious")
    logger.info(f"    D3: {len(benign_sets['D3']):,} benign + "
                f"{sum(len(v) for v in d3_malicious.values()):,} malicious "
                f"({len(d3_malicious)} families)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
