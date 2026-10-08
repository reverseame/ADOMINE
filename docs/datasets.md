# Datasets

Everything the framework consumes lives under `dataset/`. The raw sources are
Tranco and DGArchive; from them, `tools/build_datasets.py` materialises the
three evaluation splits that models and detectors actually read.

## Raw sources

### Tranco (benign domains)

- `dataset/top-1m.csv` - primary Tranco snapshot, the benign source of the
  splits.
- `dataset/tranco_1M_2026.03.24/` - dated full snapshot directory.
- `dataset/tranco_1M_2026.03.24.zip` - original archive.
- `dataset/tranco_KW6LW.csv` - Tranco list identified by its short ID.

`top-1m.csv` is byte-identical to the first 1,000,000 rows of
`tranco_KW6LW.csv` and to the copy inside `tranco_1M_2026.03.24/`. The source
is the Tranco list snapshot of 2026-03-24.

### DGArchive (malicious domains)

- `dataset/dgarchive/` - the 58 per-family CSVs with at least 50,000 samples
  each, the malicious source of the splits.
- `dataset/dgarchive_all/` - all 137 families, unfiltered.
- `dataset/dgarchive_raw/` - untouched raw downloads.

### Prefix and suffix list

- `dataset/fixes.md` - the 5,000 most frequent prefixes and suffixes for
  domains, used by the GWDGA and SADGA models. Obtained from
  https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103.

## Dataset splits (D1 / D2 / D3)

Three disjoint datasets, each with a distinct role:

- **D1** - training input for the adversarial DGA models.
- **D2** - training input for the detectors.
- **D3** - control groups (benign and malicious reference samples) used as
  evaluation baselines, i.e. the test set.

The partition is **materialised on disk** by `tools/build_datasets.py` under
`dataset/splits/`. `core/data_splits.py` reads those files and is the only
module that touches them; no model or detector opens the raw Tranco or
DGArchive sources any more.

### Files

| File | Role | Domains |
| ---- | ---- | ------- |
| `D1_benign.csv` | adversarial-model training, benign | 100,000 |
| `D1_malicious.csv` | adversarial-model AGD seeds, 58 families | 100,000 |
| `D2_benign.csv` | detector training, benign | 100,000 |
| `D2_malicious.csv` | detector training, malicious, 58 families | 100,000 |
| `D3_benign.csv` | benign control | 100,000 |
| `D3_malicious_qakbot_dga.csv` | malicious control, arithmetic | 100,000 |
| `D3_malicious_rovnix_dga.csv` | malicious control, arithmetic | 100,000 |
| `D3_malicious_suppobox_dga.csv` | malicious control, dictionary | 100,000 |
| `D3_malicious_dyre_dga.csv` | malicious control, hash | 100,000 |
| `D3_malicious_conficker_dga.csv` | malicious control, arithmetic | 100,000 |
| `D3_malicious_cryptolocker_dga.csv` | malicious control, arithmetic | 100,000 |
| `D3_malicious_banjori_dga.csv` | malicious control, permutation | 100,000 |
| `D3_malicious_symmi_dga.csv` | malicious control, pronounceable | 100,000 |
| `D3_malicious_gozi_dga.csv` | malicious control, dictionary | 100,000 |
| `D3_malicious_bamital_dga.csv` | malicious control, hash | 100,000 |
| `manifest.json` | build seed, timestamp, sources, per-file sha256 | |

One domain per line, lower-cased, no header.

### How the partition is built

- **Benign**: drawn at random from the whole Tranco list rather than by rank,
  which would bias the sample towards short, popular names.
- **Malicious**: reservoir sampling over every family file, so the draw is
  uniform over the whole file and no file is ever loaded whole. D1 and D2 take
  an equal share of each of the 58 families (1,724 each, the remainder spread
  over the first families); D3 takes a full-size sample of each control
  family, which cover the distinct generation schemes and allow a per-family
  comparison.
- **Uniqueness is global**, not per family: DGArchive families overlap, so a
  domain already taken by an earlier family is skipped.
- **Label conflicts**: the Tranco list is known to contain malicious domains
  (Le Pochat et al., 2019), so the malicious half is drawn first and any Tranco
  entry already taken as malicious is excluded from the benign half. This
  removes the conflicts with the domains actually drawn, not every Tranco entry
  that appears somewhere in DGArchive.
- **Reproducible**: one RNG per source, all seeded from `--seed` (default 42),
  so the result does not depend on file order.

Before writing, the build verifies that the three splits share no domain, that
no split holds duplicates, that the D3 families are disjoint, and that no
domain carries both labels.

### Rebuilding

```bash
python3 tools/build_datasets.py --force
```

Options: `--seed`, `--benign-per-dataset`, `--malicious-per-dataset`,
`--d3-per-family`, `--out`. A rebuild invalidates every cached weight file:
delete `models/*/weights/` and `detectors/*/weights/` and retrain.

### Adding control families without rebuilding

One reservoir per family feeds D1, D2 and D3, so adding a family to the full
build would redraw that family's D1 and D2 share and invalidate every trained
model. The incremental mode avoids that:

```bash
python3 tools/build_datasets.py --add-d3-families conficker_dga.csv gozi_dga.csv
```

Each new family gets its own RNG stream (`<seed>:<family>:d3`), every domain
already present in any split is excluded, and only the new
`D3_malicious_<family>.csv` files and the manifest change (the manifest records
the additions under `d3_families_added`). The nine files of the original build
stay byte-identical, so cached weights remain valid; only the manifest hash
stamped in `run_metadata` changes, which the next full `main.py` run refreshes.

### Loading helpers

`core/data_splits.py` exposes:

- `load_tranco(split) -> list[str]` - benign domains of `D1`, `D2` or `D3`.
- `load_dgarchive(family, split="D3") -> list[str]` - malicious domains of one
  control family (`dyre_dga.csv` and the other nine).
- `load_d2_malicious_training(total=100_000, seed=42) -> list[str]` - the
  detectors' malicious half. All nine detectors call it with the defaults, so
  they train on the identical sample.
- `load_malicious_seeds(total=None, seed=42) -> list[str]` - real AGDs from the
  D1 malicious split, for the models that mutate or seed from known malicious
  domains (MaskDGA, FGSM-based DGA, GADGA, DnGAN, CLETer,
  GeometricPerturbationDGA, WGAN-based DGA, WordDGA). `total` takes a
  deterministic subsample.
- `load_manifest() -> dict` - the build manifest.

### Validation on import

Importing `core.data_splits` checks that every file listed in the manifest is
present and that its sha256 still matches, warning when a split was modified
after the build. It reads only the manifest and the split files, so it costs
milliseconds.

## Family list

`MaliciousDGAModel` is wired in `main.py` for the ten D3 control families:

- Dyre (`dyre_dga.csv`), hash-based
- Suppobox (`suppobox_dga.csv`), dictionary-based
- Qakbot (`qakbot_dga.csv`), arithmetic
- Rovnix (`rovnix_dga.csv`), arithmetic
- Conficker (`conficker_dga.csv`), arithmetic
- Cryptolocker (`cryptolocker_dga.csv`), arithmetic
- Banjori (`banjori_dga.csv`), permutation
- Symmi (`symmi_dga.csv`), pronounceable
- Gozi (`gozi_dga.csv`), dictionary-based
- Bamital (`bamital_dga.csv`), hash-based
