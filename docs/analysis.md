# Analysis

Three analysis pipelines run on the generated samples. Each lives in its own
subdirectory under `core/analysis/` and writes JSON outputs under
`my_eval_workspace/analysis/<pipeline>/`.

## Statistical (`core/analysis/statistical/`)

Per-domain lexical features, aggregated per model.

Implemented metrics (one file each):

- `shannon_entropy.py` - Shannon entropy of the character distribution.
- `domain_length.py` - character length of the domain.
- `vowel_ratio.py` - vowels ÷ total characters.
- `consonant_ratio.py` - consonants ÷ total characters.
- `digit_ratio.py` - digits ÷ total characters.
- `unique_chars.py` - unique characters ÷ total characters.
- `consecutive_consonants.py` - longest consecutive-consonant run.

The entry point is `core/analysis/statistical/general.py:evaluate_statistical`.
It honors the `sld_only` flag from `Framework.set_analysis_mode`: when `True`,
metrics are computed on the SLD only; otherwise on the full domain.

## Detection (`core/analysis/detection/`)

For each `(detector, model)` pair, computes evasion and classification metrics
and writes one JSON per pair under
`my_eval_workspace/analysis/detection/<detector>/<model>_evasion.json`.

Entry point: `core/analysis/detection/general.py:evaluate_detection`.

### Ground-truth awareness

Each model carries an `is_malicious` flag (see `docs/architecture.md`). It is
`True` for every adversarial DGA and every malicious control, and `False` for
the benign control (`BenignDomainsModel`). `Framework.run_detection_analysis`
reads the flag from the registered model and passes it to `evaluate_detection`
as `are_malicious`, which sets the ground truth for the whole input set:

- `are_malicious=True`: the set is malicious. A domain the detector labels
  benign is a successful evasion (counted as FN); a domain it labels malicious
  is caught (TP). TN and FP stay 0.
- `are_malicious=False`: the set is benign. A domain the detector labels benign
  is correct (TN); a domain it labels malicious is a false alarm (FP). TP and
  FN stay 0.

### Outputs

The JSON holds the confusion-matrix counts (`TP`, `TN`, `FP`, `FN`) and the
derived metrics (`Accuracy`, `Precision`, `Recall`, `F1 score`, `FPR`, `TPR`,
`MCC`, `Kappa`), plus:

When the input set is empty, `total_domains` and the confusion counts are 0 and
every rate and metric is `null`, so an empty set can never be read as a 0%
evasion rate.

- `evaded_domains` - number of domains the detector passed as benign. For a
  malicious set this is the evasion count (= FN); for the benign control it is
  the count the detector correctly left alone (= TN).
- `evasion_rate_percent` - `evaded_domains / total_domains` as a percentage.

For the benign control the ground truth has a single class, so the metrics
that need both classes (`Precision`, `Recall`, `TPR`, `F1 score`, `MCC`,
`Kappa`) are not informative and report 0; the meaningful outputs are
`Accuracy`, `FPR`, `TN`, and `FP`. `Kappa` follows the standard
expected-agreement formula, which collapses to 0 whenever the ground truth
has a single class.

## Time (`core/analysis/time/`)

Per-model training, generation, and per-domain timing; per-detector training
and inference timing.

Entry point: `core/analysis/time/general.py:evaluate_time`. The framework
merges the new run's times with any pre-existing JSON so partial reruns don't
zero out unchanged columns.

For models whose `fit()` can train or load cached weights, the JSON also
records `trained_from_scratch`: `true` when the run trained, `false` when it
loaded cached weights. The merge protects real numbers. Once a run has written
a training time with `trained_from_scratch: true`, a later cache-hit run
(`false`) leaves that time and flag in place instead of overwriting them with
the weight-load time. A cache-hit run writes the load time only when no real
training time has been recorded yet.

Detectors get the same treatment through
`core/analysis/time/general.py:evaluate_detector_time`. At the end of the
detection analysis the framework writes one
`analysis/time/<Detector>_time_eval.json` per detector with
`training_time_seconds`, `trained_from_scratch`, `inference_time_seconds`
(wall time of the `detect` calls, accumulated over every model scored in the
run), `total_domains_scored`, and `inference_time_per_domain_seconds`. The
cache-hit protection above applies to detector training times too. Detection
JSONs carry no timing fields, so they stay seed-deterministic; all
wall-clock numbers live under `time/`.

Both accumulators cover every `detect` call of the run. With the SLD-only pass
of `main.py` (Stage 4) the JSON written last therefore spans the two passes:
`inference_time_seconds` is the wall time of both and `total_domains_scored` is
twice the number of evaluated domains. `inference_time_per_domain_seconds`
stays a valid average over all scored inputs and is the figure to report.

## SLD-only vs full-domain mode

The framework state `analyze_sld_only` is set at construction
(`Framework(sld_only=...)`), flippable mid-run via
`set_analysis_mode(sld_only)`, and consumed by the statistical and detection
pipelines.

SLD extraction is public-suffix aware: `core/sld.py:extract_sld` uses
`tldextract` with its bundled suffix snapshot (no network fetch), so
`example.co.uk` yields `example` and `mail.google.com` yields `google`. A
bare label passes through unchanged; when the input has no registrable part
the first label is used. Only the ICANN section of the Public Suffix List is
applied, so the SLD is the second-level label by DNS hierarchy: in
`poniutnovarauga.ddns.net` the SLD is `ddns`, not the generated label. Some
families generate the third or fourth label under a dynamic DNS or free
hosting provider (Symmi under `ddns.net`, Bamital under `co.cc` and `cz.cc`);
in SLD-only mode the detectors then see the provider's label, which is a
property of those families, not of the extraction. Their SLD-only figures
must be read that way.

In SLD-only mode the detection pipeline strips every input domain to its SLD
before calling the detectors (which stay trained on full domains) and writes
`<model>_evasion_sld.json` next to the full-domain `<model>_evasion.json`,
so both modes can be reported from one workspace. The statistical pipeline
does the same: `<Model>_statistical_eval_sld.json` next to
`<Model>_statistical_eval.json`. `main.py` runs the full-domain pass first and
then repeats statistical and detection analyses in SLD-only mode. Every detection JSON
carries an `"sld_only"` boolean naming its mode.

The detectors are trained on full domains, and the recurrent ones can react to
the missing dot and TLD at the end of the sequence as a DGA signal in itself.

### Oracle input of the black-box attacks

CLETer, WordDGA and PKDGA query the detector during the attack (influence
scores, evasion check, reinforcement rewards). Following their papers, which
strip the TLD in preprocessing and train their target classifiers on the SLD,
the query carries the SLD alone; the TLD is appended afterwards to form the
returned domain. The framework's detectors are trained on full domains, so
the two modes measure two different things for these three models:

- SLD-only mode: oracle input and evaluated input coincide. This is the
  measurement comparable with the papers.
- Full-domain mode: the attack was tuned against the SLD view and the
  evaluated detector sees the full domain. This is a transfer setting, and a
  realistic one: an attacker does not control whether the deployed detector
  consumes the SLD or the full domain.

TITAN DGA queries and evaluates with the full domain, as its paper does. The
remaining models do not query a detector at generation time.
