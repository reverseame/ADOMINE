# Reproducibility

How to reproduce a run end-to-end.

## End-to-end run

```bash
python3 main.py
```

That single command:

1. Fits every adversarial model and every control model.
2. Either loads pre-generated samples from `my_eval_workspace/samples/` or
   generates `n = 100_000` fresh ones per model.
3. Runs statistical, time, and detection analyses, writing JSONs under
   `my_eval_workspace/analysis/`.

To force a clean run, delete `my_eval_workspace/` (or just `samples/` for a
sample-level redo) and any cached model weights inside `models/<name>/` and
`detectors/<name>/`.

## Seeding

`main.py` defines a module-level `SEED = 42` and calls `seed_everything(SEED)`
(from `core/seeding.py`) as the first line of `main()`, before any model is
built, fit, or sampled. `seed_everything` seeds the Python `random`, NumPy
(`np.random`), and TensorFlow (`tf.random.set_seed`) global RNGs. The same value
is passed to `Framework(seed=SEED)`, which stamps it inside the `"run_metadata"`
object of every analysis JSON it writes. To run under a different seed, change
the `SEED` constant in `main.py`.

## Run metadata

Every analysis JSON carries a top-level `"run_metadata"` object: UTC
timestamp, seed, Python / TensorFlow / NumPy / tldextract versions, a sha256
of the Tranco CSV, and a structural fingerprint of the DGArchive directory
(`core/run_metadata.py`, collected once per `Framework` construction). A
number in the paper traces to a JSON, and the JSON names the inputs and
toolchain that produced it. Field list in
[`workspace.md`](workspace.md#run-metadata).

## Training times

Every model that trains caches its weights under `models/<name>/weights/` and
short-circuits `fit()` when they exist, recording only the weight-load time,
flagged `trained_from_scratch: false` in the time JSON. To record real training
times, delete the cached weights of that model and run once. That run writes the real time with `trained_from_scratch: true`, which
later cache-hit runs preserve. The detectors cache weights the same way; delete
the weights under `detectors/<name>/` to retrain them.

## Python and dependencies

Supported Python: 3.9 to 3.12, bounded by the TensorFlow pin. Install the
dependencies with:

```bash
pip install -r requirements.txt
```

The pins in `requirements.txt` fix the exact versions to install. TensorFlow
is required for the detectors and for DeepDGA.
A run without TensorFlow aborts at detector fitting with a `RuntimeError`
instead of producing fallback numbers.

## Dataset snapshots

Models and detectors read only the materialised splits under
`dataset/splits/`, built once from the raw sources by
`tools/build_datasets.py`. See [`datasets.md`](datasets.md).

- Raw benign source: `dataset/top-1m.csv` (Tranco snapshot of 2026-03-24; a
  dated full copy is kept under `dataset/tranco_1M_2026.03.24/`).
- Raw malicious source: `dataset/dgarchive/`, the 58 families with at least
  50,000 samples.
- `dataset/fixes.md`, the 5,000 most frequent prefixes and suffixes, consumed
  by `GWDGAModel` and `SADGAModel`, obtained from
  https://gist.github.com/cnicodeme/b267e45115a77c474e1cf7a544d98103.

To rebuild the splits:

```bash
python3 tools/build_datasets.py --force
```

A rebuild changes what every model and detector trains on, so delete the
cached weights under `models/*/weights/` and `detectors/*/weights/` and the
samples under `my_eval_workspace/samples/` before the next run. Every analysis
JSON records which partition produced it through `dataset_manifest_sha256`.
