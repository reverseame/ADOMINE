# Detectors

Detectors live under `detectors/`, one per subdirectory, and inherit from
`core.detector.Detector`. They are evaluated against every registered model in
the detection-analysis stage, in full-domain mode and in SLD-only mode. All
of them train on full domains; what that implies for the SLD-only figures,
especially for the recurrent networks, is in
[`analysis.md`](analysis.md#sld-only-vs-full-domain-mode).

Each detector's `fit()` records `training_time` and sets `trained_from_scratch`
(`true` on a real training run, `false` when it loads cached weights). Every
detector caches its weights under `detectors/<name>/weights/` and
short-circuits `fit()` when the file exists; delete it to retrain.

All detectors require TensorFlow. `fit()` and `detect()` raise `RuntimeError`
when it is not installed; there is no fallback path.

## Interface contract

`detect(domains: list[str]) -> list[tuple[str, bool]]`. The bool is
`True` if the detector classifies the domain as **benign**, `False` if
**malicious**.

## Training data shape

All detectors share the same wiring through `core/data_splits.py`:

- Malicious half: `load_d2_malicious_training()` returns the D2 malicious
  split (`dataset/splits/D2_malicious.csv`, 100,000 domains), already built as
  an equal-per-family draw over the 58 family files. Every detector calls it
  with the defaults, so they all train on the identical sample regardless of
  call order.
- Benign half: read `TRANCO_D2_BENIGN` (`dataset/splits/D2_benign.csv`) and
  truncate to match the malicious count. Both halves hold 100,000 domains, so
  the training set is balanced without discarding anything.

See [`datasets.md`](datasets.md) for how the splits are built.

## RAMPAGE detectors

Seven detectors reproduce the seven best models of the RAMPAGE evaluation
(Pelayo-Benedet, Rodríguez and Gañán, "RAMPAGE: A Software Framework To Ensure
Reproducibility in Algorithmically Generated Domains Detection", Expert Systems
With Applications; code at https://github.com/reverseame/RAMPAGE). They are
the seven networks RAMPAGE combined in its meta-model. The meta-model itself,
a logistic regression over the seven probabilities, is not part of this
framework: each network is evaluated as an independent detector.

| Directory | Class | Label in the RAMPAGE paper | Original paper | Network | Optimizer |
|---|---|---|---|---|---|
| `woodbridge_lstm/` | `WoodbridgeLSTM` | LSTM (Woodbridge et al., 2016) | Woodbridge et al., 2016 | Embedding(256, 128), LSTM(128), Dropout(0.5), Dense(1), sigmoid | Adam, lr 1e-3, weight decay 1e-3 |
| `yu_lstm/` | `YuLSTM` | LSTM (Yu et al., 2017) | Yu et al., 2017 | Embedding(256, 128, mask_zero), LSTM(128), Dropout(0.5), Dense(100), Dense(1), sigmoid | Adam, lr 1e-3, weight decay 1e-3 |
| `tweet2vec/` | `Tweet2Vec` | MCU (Yu et al., 2018) | Dhingra et al., 2016, via Yu et al., 2018 | Embedding(128, 128), Bidirectional LSTM(64), Dense(1), sigmoid | Adam |
| `tweet2vec2/` | `Tweet2Vec2` | MIT (Yu et al., 2018) | Vosoughi et al., 2016, via Yu et al., 2018 | Embedding(128, 128), Conv1D(128, k3), MaxPool(2), LSTM(64), Dense(1), sigmoid | Adam |
| `dbd/` | `DBD` | DBD (Vinayakumar et al., 2019) | Vinayakumar et al., 2019 | Embedding(256, 128), Conv1D(64, k5), MaxPool(4), LSTM(70), Dense(1), sigmoid | Adam |
| `zhang/` | `Zhang` | CNN (Berman, 2019) | Zhang et al., 2015, via Yu et al., 2018 | Embedding(128, 128), 2 x [Conv1D(128), ThresholdedReLU, MaxPool(2)], Dense(64), Dropout(0.5), Dense(1), sigmoid | Adam |
| `yang_cnn/` | `YangCNN` | Parallel CNN (Yu et al., 2018), in the meta-model experiments | Yang et al., 2018 | Embedding(256, 128), Conv1D(128, k5), MaxPool(4), Dropout(0.5), Dense(128), Dropout(0.5), Dense(1), sigmoid | RMSprop |

Class names, directory names and layer settings follow the RAMPAGE code
(`classifiers/<Author>/<class>.py`). Two labels of the paper do not name the
network that produced its numbers; the decisions are recorded in
[`assumptions.md`](assumptions.md#pelayo-benedet-et-al-2025-rampage-detectors).

Shared training setup, taken from RAMPAGE:

- Input: each character encoded as `ord(c) - 33` (RAMPAGE `commonData`),
  post-padded to 75 positions. RAMPAGE pads to 70; 75 is the length of the
  original detectors and covers every domain of the splits and of the
  generated samples without truncation.
  `YuLSTM` masks the padding (`mask_zero=True`), the one departure from the
  RAMPAGE code; without it the LSTM runs over the padding positions and
  reaches the same state for every input.
- Label 1 = DGA. `detect()` returns benign when P(DGA) <= 0.5.
- Batch size 50, up to 500 epochs, early stopping on validation accuracy with
  patience 10 and the best weights restored. The last 20 percent of a seeded
  permutation of the training set is held out for validation; the permutation
  is fixed (`SPLIT_SEED = 42`), so the seven detectors see the same split.
- Loss: binary cross-entropy.

## Original detectors (`detectors/lstm/`, `detectors/cnn/`)

The two original detectors of the framework. They differ from the RAMPAGE
implementations in alphabet (39 symbols plus padding instead of raw character
codes), optimizer and schedule (100 epochs, no early stopping).

### LSTM (`detectors/lstm/`)

Class `LSTMDetector`, display name `LSTM`. Woodbridge-style LSTM:
Embedding(40, 128), LSTM(128), Dropout(0.5), Dense(1); rmsprop, batch 32,
100 epochs.

### CNN (`detectors/cnn/`)

Class `CNNDetector`, display name `CNN`. Character-level CNN:
Embedding(40, 128), Conv1D(128, k5), GlobalMaxPooling1D, Dense(64),
Dropout(0.5), Dense(1); adam, batch 64, 100 epochs.

It is also the default black-box oracle that CLETer, WordDGA, PKDGA and
TITANDGA query during generation.

## Adding a new detector

1. Create `detectors/<your_detector>/<your_detector>.py`.
2. Inherit from `Detector` and implement `fit` and `detect`.
3. Make `fit()` fully self-contained.
4. Register in `main.py` with `fw.register_detector(...)`.
