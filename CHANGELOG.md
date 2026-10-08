# Changelog

All notable changes to this project will be recorded here.
Dates follow `YYYY-MM-DD`.

## 2027-10-08 - Version 1.0 release

- Version 1.0 release with all functionality

## 2026-09-21 - RAMPAGE detectors

- Seven more detectors under `detectors/`, the seven best models of the
  RAMPAGE evaluation, trained on D2 like the two original ones. Details in
  `docs/detectors.md` and `docs/assumptions.md`.

## 2026-09-04 - D1/D2/D3 datasets

- `tools/build_datasets.py` builds the three disjoint splits into
  `dataset/splits/` (100,000 benign and 100,000 malicious each; D3 additionally
  holds one 100,000-domain file per control family). Random benign draw,
  reservoir sampling per family, global uniqueness, and no domain carrying both
  labels. A manifest records the seed and the sha256 of every file.
- `core/data_splits.py` now reads those files and is the only module that
  touches the datasets; no model or detector opens Tranco or DGArchive
  directly.
- `core/run_metadata.py` stamps `dataset_manifest_sha256`, `dataset_build_seed`
  and the sources into every analysis JSON, replacing the previous per-source
  hashes.

## 2026-09-03 - Model fixes and framework guards

- Bounded rejection sampling with logged fallbacks in every generator that
  could loop forever or emit invalid domains; no literal placeholder domains
  remain.
- RFC 1034/1035 validation of generated labels (NDG, PKDGA, WGAN-based DGA),
  lower-cased output where the paper's encoding allows upper case, and a
  framework-wide TLD convention (`core/tld.py`) for the eight models whose
  paper generates only the SLD.
- `main.py` reports the generation stage in `samples/generation_report.json`,
  reuses a sample cache only when it is complete, and repeats the statistical
  and detection analyses in SLD-only mode.
- `core/seeding.py` seeds SentencePiece with the API of sentencepiece 0.2.0 and
  exposes `get_seed()`; Word2Vec is now seeded in CDGA and TLVDGA.

## 2026-06-01 - Base codebase

- Initial documented snapshot.
