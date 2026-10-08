# Workspace

All dynamic output (samples and analysis JSONs) is written under the
workspace directory. Its path is set when the framework is constructed:

```python
fw = Framework(workspace_dir="my_eval_workspace")
```

`main.py` uses `my_eval_workspace/`, which is the repository's default.

## Layout

```
my_eval_workspace/
  samples/                                  # raw generated domains, one file per model
    DeepDGA_samples.txt
    CharBot_samples.txt
    ...
  analysis/
    statistical/                            # per-model lexical metrics
      DeepDGA_statistical_eval.json
      DeepDGA_statistical_eval_sld.json         # SLD-only mode
      CharBot_statistical_eval.json
      ...
    time/                                   # per-model and per-detector timing
      DeepDGA_time_eval.json
      LSTM_time_eval.json
      CNN_time_eval.json
      Woodbridge_LSTM_time_eval.json
      Yu_LSTM_time_eval.json
      ...                                     # one per detector
    detection/                              # per-detector × per-model evasion
      lstm/
        deepdga_evasion.json                # full-domain mode
        deepdga_evasion_sld.json            # SLD-only mode, when run
        charbot_evasion.json
        ...
      cnn/
        deepdga_evasion.json
        ...
      woodbridge_lstm/                      # the seven RAMPAGE detectors
      yu_lstm/
      tweet2vec/
      tweet2vec2/
      dbd/
      zhang/
      yang_cnn/
```

Detection JSONs carry an `"sld_only"` boolean naming the mode they were
produced in. The two modes write to different file names, so they coexist;
see [`analysis.md`](analysis.md#sld-only-vs-full-domain-mode).

## Run metadata

Every analysis JSON (under `statistical/`, `time/`, and `detection/`) carries a
top-level `"run_metadata"` object identifying the run that wrote it:

- `timestamp_utc` - ISO 8601 time the framework was constructed.
- `seed` - the RNG seed passed to `Framework(seed=...)`.
- `python_version`, `tensorflow_version`, `numpy_version`,
  `tldextract_version` - interpreter and dependency versions.
- `dataset_manifest_sha256` - content hash of `dataset/splits/manifest.json`,
  which itself carries the sha256 of every D1/D2/D3 file, so this single value
  pins the whole partition.
- `dataset_build_seed`, `dataset_built_utc` - seed and timestamp of the
  `tools/build_datasets.py` run that produced the splits.
- `dataset_benign_source`, `dataset_malicious_source` - raw sources the splits
  were drawn from.

For time JSONs, values preserved by the merge (e.g. a real training time from
an earlier run) can predate the run named in `run_metadata`; the metadata
identifies the run that last wrote the file. See
[`reproducibility.md`](reproducibility.md#run-metadata).

## Sample caching behavior

`main.py` checks for `samples/<Name>_samples.txt` before generating. The file
is reused only when it holds at least the requested number of domains; a
shorter file (left by a model that could not fill its quota in an earlier
run) is regenerated with a warning. When the cache is reused,
`generation_time` is **not** updated. To force regeneration, delete the file.

After the generation stage `main.py` writes `samples/generation_report.json`:
per model, `requested`, `obtained`, `source` (`cache` or `generated`) and
`short` (true when the model returned fewer domains than requested). A short
sample is also logged as a warning. Results of a short model rest on fewer
domains than the others and must be read with that in mind.

Caching is keyed on the model's display name (e.g., `Rovnix_samples.txt`).
Renaming a model orphans its old cache file: the next run silently
regenerates samples under the new name and never reads the old file again.
When renaming a model, delete or rename the old
`<OldName>_samples.txt` to keep the cache and the model list in sync.

## Time JSON merge behavior

`run_time_analysis` reads the existing `*_time_eval.json` if present and merges
the new values in. This lets you re-run only one model's generation step
without zeroing out the others.
