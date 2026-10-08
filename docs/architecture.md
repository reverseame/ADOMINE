# Architecture

Three abstractions and one orchestrator. Every adversarial DGA and every
detector can be added or removed without touching the orchestration code.

## Top-level layout

```
core/
  adversarial_model.py   # AdversarialModel ABC
  data_splits.py         # D1/D2/D3 split loaders (dataset/splits/)
  tld.py                 # framework TLD convention for SLD-only models
  detector.py            # Detector ABC
  framework.py           # Framework orchestrator
  seeding.py             # seed_everything(): seeds random / numpy / tf RNGs
  analysis/
    statistical/         # Lexical metric evaluators + general.py
    detection/           # Detector-side evaluators (evasion, TPR/FPR, etc.)
    time/                # Training/generation time evaluators
models/                  # One subdirectory per adversarial DGA
detectors/               # One subdirectory per detector
dataset/                 # Tranco + DGArchive + training files raw inputs
main.py                  # Entry point that wires it all together
```

## Abstractions

### `AdversarialModel` (`core/adversarial_model.py`)

Every adversarial DGA model inherits from this ABC and must implement:

- `fit() -> None` - self-contained training or initialization. Takes no
  arguments. Each model owns its dataset paths, hyperparameters, and artifact
  locations, so the framework stays decoupled from per-model details.
- `generate_domain() -> str` - produce a single domain.
- `generate_domains(n: int) -> list[str]` - produce `n` domains.

Models also expose runtime attributes used by the time analysis:

- `training_time` - wall time of `fit()`. A model whose `fit()` can either
  train or load cached weights sets this itself; otherwise `main.py` times the
  `fit()` call and fills it in.
- `generation_time` - set inside `Framework.generate_all` (or, in `main.py`'s
  inline path, in the sample-generation loop).
- `trained_from_scratch` - `True` when `fit()` ran a real training loop,
  `False` when it loaded cached weights, `None` for models that never train.
  Defaults to `None` in `AdversarialModel.__init__`.

Every model also carries `is_malicious`, a ground-truth flag consumed by the
detection analysis. It defaults to `True` in `AdversarialModel.__init__` and is
overridden to `False` by `BenignDomainsModel`. See `docs/analysis.md` for how
the detection pipeline uses it.

### `Detector` (`core/detector.py`)

Detectors mirror the model interface:

- `fit() -> None` - self-contained, just like models.
- `detect(domains: list[str]) -> list[tuple[str, bool]]` - return a list of
  `(domain, is_benign)` tuples. **`True` means benign, `False` means malicious.**

Detectors also carry `training_time` and `trained_from_scratch`, set inside
`fit()` with the same meaning as on models, plus `inference_time` and
`inference_domains`, accumulated by the framework across every `detect`
call of the run.

### `Framework` (`core/framework.py`)

Single orchestrator that:

1. Registers models (`register_model`) and detectors (`register_detector`).
2. Creates the workspace directory tree and owns its paths.
3. Coordinates generation (`generate_all`) or accepts pre-generated samples
   from the caller - `main.py` uses the latter so cached `.txt` samples can be
   reused.
4. Delegates to the three analysis pipelines:
   - `run_statistical_analysis` → `core/analysis/statistical/general.py`
   - `run_time_analysis` → `core/analysis/time/general.py`
   - `run_detection_analysis` → `core/analysis/detection/general.py`
5. Carries a single piece of analysis state, `analyze_sld_only`, set at
   construction (`Framework(sld_only=...)`) and flippable mid-run via
   `set_analysis_mode(sld_only=...)`.
6. Takes a `seed` (default 42) and stamps a top-level `"run_metadata"` object
   (timestamp, seed, dependency versions, dataset hashes; collected once by
   `core/run_metadata.py`) into every analysis JSON. See
   `docs/reproducibility.md`.

## Control flow (current `main.py`)

1. Instantiate `Framework(workspace_dir="my_eval_workspace")`.
2. Instantiate every adversarial model (DeepDGA, CharBot, Deception, MaskDGA, HMM-based DGA, PCFG-based DGA, FGSM-based DGA, GADGA, WGAN-GP DeepDGA, Khaos, DnGAN, ShadowDGA, NDG, CLETer, Geometric Perturbation DGA, CDGA, GWDGA, WGAN-based DGA, ReplaceDGA, PKDGA, WordDGA, TLVDGA, TITAN DGA, SADGA),
   plus the malicious controls (`MaliciousDGAModel` for Dyre, Suppobox, Qakbot,
   Rovnix, Conficker, Cryptolocker, Banjori, Symmi, Gozi, Bamital) and the
   benign control (`BenignDomainsModel`).
3. Call `model.fit()` for each, timing the call externally.
4. Register all models with the framework.
5. For each model: either load pre-existing samples from
   `workspace/samples/<Name>_samples.txt`, or generate `n` fresh ones and write
   them to that path.
6. Run statistical analysis, then time analysis.
7. Instantiate and `fit()` the detectors: the two original ones (LSTM, CNN)
   and the seven best models of the RAMPAGE evaluation (Woodbridge LSTM, Yu
   LSTM, Tweet2Vec, Tweet2Vec2, DBD, Zhang, Yang CNN). Register them and run
   detection analysis.
8. Switch to SLD-only mode (`set_analysis_mode(sld_only=True)`) and repeat the
   statistical and detection analyses; results land in `*_sld.json` files next
   to the full-domain ones.

## Decoupling guarantees

- A new model can be added by dropping a subdirectory under `models/` and
  registering it in `main.py`. No change to `core/` is required.
- Same for detectors under `detectors/`.
- Analysis modules consume plain `list[str]` inputs and produce JSON outputs.
  They do not import models or detectors.
