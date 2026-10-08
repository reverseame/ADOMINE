# ADOMINE

**A**dversarial **D**omain generati**O**n algorith**M**s compar**I**so**N** fram**E**work

![Python](https://img.shields.io/badge/python-3.9--3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)

ADOMINE is a reproducible framework for evaluating and comparing **adversarial
Domain Generation Algorithms (DGAs)**: generators built to produce
algorithmically generated domains (AGDs) that evade machine-learning DGA
detectors. It re-implements 24 adversarial DGAs from the literature behind a
single interface, trains 9 deep-learning detectors on a common dataset, and
scores every generator against every detector with the same lexical,
detection and timing metrics, so results across papers become directly
comparable.

- [Getting started](#getting-started)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [Extending ADOMINE](#extending-adomine)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Citing](#citing)
- [License](#license)

## Getting started

### Requirements

- Python 3.9 to 3.12 (bounded by the TensorFlow pin).
- The pinned dependencies in `requirements.txt` (TensorFlow 2.18, PyTorch 2.3,
  NumPy, gensim, SentencePiece, tldextract, wordninja, NLTK). TensorFlow is
  mandatory: the detectors raise instead of falling back.
- A GPU is strongly recommended. Training all 24 generators and 9 detectors
  from scratch and generating 3.5 million domains is a long run on CPU.
- Access to the two raw data sources below. They are not redistributed with
  this repository.

### Installation

```bash
git clone https://github.com/reverseame/ADOMINE.git
cd ADOMINE
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data

The whole `dataset/` directory is git-ignored. Create it and place the raw
sources as follows:

| Path | Content | Source |
| --- | --- | --- |
| `dataset/top-1m.csv` | Tranco top-1M list (`rank,domain`) | [tranco-list.eu](https://tranco-list.eu/). The reference build uses the snapshot of 2026-03-24. |
| `dataset/dgarchive/<family>.csv` | One CSV per DGA family, one domain per line (`domain` or `date,domain`) | [DGArchive](https://dgarchive.caad.fkie.fraunhofer.de/) (Fraunhofer FKIE, access on request). The reference build uses the 58 families with at least 50,000 domains; the ten control families evaluated by `main.py` must be present: Qakbot, Rovnix, Conficker, Cryptolocker, Suppobox, Gozi, Dyre, Bamital, Banjori and Symmi. |
| `dataset/fixes.md` | Optional list of frequent domain prefixes and suffixes, one `* token` per line, used by GWDGA and SADGA | [gist](https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103). Without it both models fall back to dataset frequencies. |

Then build the D1/D2/D3 splits once. The reference partition was built in
two steps, and reproducing it requires the same two commands in this order:

```bash
python3 tools/build_datasets.py
python3 tools/build_datasets.py --add-d3-families conficker_dga.csv cryptolocker_dga.csv banjori_dga.csv symmi_dga.csv gozi_dga.csv bamital_dga.csv
```

The first command draws 100,000 benign and 100,000 malicious domains per
split (reservoir sampling over every family file, global uniqueness, no label
conflicts), writes them to `dataset/splits/` with Qakbot, Rovnix, Suppobox and
Dyre as D3 control families, and records the seed, the sources and the sha256
of every file in `dataset/splits/manifest.json`. The second command appends
the other six control families that `main.py` evaluates without touching the
files already written. `main.py` aborts with a `FileNotFoundError` if any
split file it needs is missing. Options and the validation performed before
writing are described in [`docs/datasets.md`](docs/datasets.md).

### Run

```bash
python3 main.py
```

The run is resumable: generated samples and trained weights are cached, so an
interrupted run picks up where it stopped. To force a clean run, delete
`my_eval_workspace/` and the `weights/` directories under `models/*/` and
`detectors/*/`. To change the seed, edit `SEED` in `main.py`; to evaluate a
subset of models or detectors, comment out their registration there.

## Outputs

Everything is written under `my_eval_workspace/`:

```
my_eval_workspace/
  samples/                     one <Model>_samples.txt per model, plus generation_report.json
  analysis/
    statistical/               <Model>_statistical_eval.json and *_sld.json
    time/                      <Model>_time_eval.json and <Detector>_time_eval.json
    detection/<detector>/      <model>_evasion.json and *_evasion_sld.json
```

Detection JSONs hold the confusion matrix and Accuracy, Precision, Recall,
F1, FPR, TPR, MCC and Kappa, plus the evasion count and rate. Every analysis
JSON carries a `run_metadata` object with the timestamp, seed, dependency
versions and the hash of the dataset manifest, so any number traces back to
the inputs and toolchain that produced it. The full schema is in
[`docs/workspace.md`](docs/workspace.md) and
[`docs/analysis.md`](docs/analysis.md).

## Repository layout

```
core/
  adversarial_model.py   AdversarialModel abstract base class
  detector.py            Detector abstract base class
  framework.py           Framework orchestrator
  data_splits.py         D1/D2/D3 loaders, the only module that reads dataset/splits/
  run_metadata.py        run metadata stamped into every output JSON
  seeding.py             seed_everything() for random, NumPy and TensorFlow
  sld.py, tld.py         public-suffix-aware SLD extraction and TLD convention
  analysis/              statistical/, detection/ and time/ pipelines
models/                  one directory per adversarial DGA, control or dummy model
detectors/               one directory per detector
tools/build_datasets.py  builds the D1/D2/D3 splits from Tranco and DGArchive
docs/                    design, models, detectors, datasets, analysis, assumptions
main.py                  end-to-end evaluation entry point
```

## Extending ADOMINE

A new generator or detector needs no change to `core/`:

- **Model**: create `models/<name>/<name>.py`, subclass `AdversarialModel`
  and implement `fit()`, `generate_domain()` and `generate_domains(n)`. Read
  training data through `core/data_splits.py`, write weights inside the
  model's own directory, and register the model in `main.py`.
- **Detector**: create `detectors/<name>/<name>.py`, subclass `Detector` and
  implement `fit()` and `detect(domains)`, which returns `(domain, is_benign)`
  pairs. Register it in `main.py`.

Details in [`docs/architecture.md`](docs/architecture.md).

## Reproducibility

- One module-level seed (`SEED = 42`) seeds Python, NumPy and TensorFlow
  before anything is built, and is stamped into every result.
- Dependencies are pinned to exact versions in `requirements.txt`.
- The dataset manifest pins the whole partition with per-file sha256 hashes,
  checked on import; a rebuilt partition is detected and reported.
- Models and detectors cache their weights and record whether a run trained
  from scratch or loaded a cache, so reported training times are real.
- Every departure from a reproduced paper is written down in
  [`docs/assumptions.md`](docs/assumptions.md), paper by paper.

The step-by-step procedure is in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Documentation

| Document | Content |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | Abstract base classes, orchestrator and control flow |
| [`docs/models.md`](docs/models.md) | Every adversarial, control and dummy model |
| [`docs/detectors.md`](docs/detectors.md) | Detector architectures and training setup |
| [`docs/datasets.md`](docs/datasets.md) | Raw sources, split construction and loaders |
| [`docs/analysis.md`](docs/analysis.md) | Metrics of the three pipelines and the SLD-only mode |
| [`docs/workspace.md`](docs/workspace.md) | Output layout and JSON schemas |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Reproducing a run end to end |
| [`docs/assumptions.md`](docs/assumptions.md) | Information gaps and decisions per reproduced paper |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes |

## Citing

TBD. A citation will be provided once the accompanying paper is published.

## License

ADOMINE is released under the
[GNU General Public License v3.0](LICENSE). Third-party code adapted into
individual models (for example PKDGA's `generator.py` and `rollout.py`) keeps
the attribution noted in its module documentation.
