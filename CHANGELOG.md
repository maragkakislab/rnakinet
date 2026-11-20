# Changelog

Changes to the `rnakinet` Python package (the `rnakinet-inference` and
`rnakinet-predict-halflives` commands and the packaged models).

## [Unreleased]

### Changed
- **Breaking:** the prediction table columns are now `mod_score` and
  `mod_prediction` (were `5eu_mod_score` and `5eu_modified_prediction`), as
  models may detect 5EU or 4SU. `rnakinet-predict-halflives` still accepts
  tables with the old column names.
- Padding is now determined by the model architecture, also for
  `--model-path`: v1.0 checkpoints are run on unpadded reads, as the
  packaged v1.0 models, and v2.0 checkpoints on padded reads. Custom v1.0
  checkpoints were previously run on padded reads (#14).
- Models that do not pad reads run with `--batch-size 1`, with a warning
  (larger batches failed).
- `--arch` accepts only the known architectures (`v1.0`, `v2.0`).
- Requires Python >= 3.10. Uses `lightning.pytorch` (Lightning >= 2.0)
  instead of `pytorch_lightning`. Tested on Python 3.10 to 3.14.

### Added
- `--read-ids`: predict only the reads listed in a file, read directly from
  the raw POD5 files; reports how many listed reads were found and stops if
  none are.

### Fixed
- Inference no longer crashes when no reads pass the length filter; it
  writes an empty table and warns with the filter's bounds.

### Removed
- The Snakemake training workflow, its scripts and environments,
  which moved to a separate repository.
- The `r10_5EU_v1.0` model. Its checkpoint could not be loaded (path
  without `.ckpt`, registered with the wrong architecture).
- FAST5 support code and the `ont-fast5-api`, `scipy` and `torchvision`
  dependencies.

## [2.0.1] - 2026-04-10

### Added
- `--version`.
- Default values are shown in `rnakinet-inference --help`.

### Fixed
- The `r10_5EU_v2.0` model pointed to a checkpoint file that did not exist;
  its checkpoint was renamed and replaced with the correct one.

### Changed
- Model descriptions in `--help`.

## [2.0.0] - 2026-04-10

### Added
- RNAkinet v2.0 architecture and the `r10_5EU_v2.0` model (R10, padded
  reads).
- `--model-name` to choose a packaged model (`r9_5EU_v1.0`,
  `r10_5EU_v1.0`, `r10_5EU_v2.0`), and `--model-path` with `--arch` for
  custom checkpoints.
- `--log` to write the inference parameters and the fraction of reads
  predicted modified.

### Changed
- **Breaking:** input is POD5 only, given with `--path` (files or
  directories, searched recursively). `--format` was removed; the hidden
  `--pod5-files` is deprecated in favor of `--path`.
- **Breaking:** `--kit` is deprecated in favor of `--model-name`
  (`--kit r9` maps to `r9_5EU_v1.0`, `--kit r10` to `r10_5EU_v2.0`).
- Status messages are printed to stderr.
- Depends on Lightning again (`lightning`, `torchmetrics`); looser version
  requirements; Python >= 3.8, < 3.13.

## [1.0.1] - 2025-02-15

### Added
- POD5 input: `--format pod5` (default `fast5`).

### Changed
- Requires `numpy` 1.24 (was 1.26).

## [1.0.0] - 2024-08-30

First stable release; same code as 0.1.29. The README was updated.

## [0.1.26] to [0.1.29] - 2024-08-29

### Added
- A model for the R10 chemistry: `--kit r9` (default) or `--kit r10`
  selects the packaged model, replacing `--checkpoint`.

### Fixed
- The R10 model's architecture, the `--kit` default and the inference
  script (0.1.27, 0.1.28).

### Changed
- Requires `numpy` 1.26 (0.1.29).

## [0.1.24] - 2023-12-22

### Added
- `--use-cpu` to run inference without a GPU.

## [0.1.23] - 2023-12-22

### Changed
- Requires PyTorch >= 2.0, <= 2.2, and Python >= 3.8, < 3.11.

## [0.1.20] to [0.1.22] - 2023-12-05

### Changed
- Inference runs on plain PyTorch; PyTorch Lightning is no longer a
  dependency. Python >= 3.7, < 3.11.

## [0.1.19] - 2023-11-07

### Changed
- Inference no longer writes Lightning logger output.

## [0.1.17] and [0.1.18] - 2023-11-06 to 2023-11-07

### Added
- The trained model is included in the package and used by default;
  `--checkpoint` is optional.

### Fixed
- The Python version requirement (>= 3.7, <= 3.10) and the package version
  number.

## [0.1.0] to [0.1.16] - 2023-10-31

Initial releases (packaging iterations on the same day). `rnakinet-inference`
predicts 5EU-labeled reads from FAST5 files (`--checkpoint` required), and
`rnakinet-predict-halflives` estimates transcript half-lives from the
predictions.

[Unreleased]: https://github.com/maragkakislab/rnakinet/compare/v2.0.1...HEAD
[2.0.1]: https://pypi.org/project/rnakinet/2.0.1/
[2.0.0]: https://pypi.org/project/rnakinet/2.0.0/
[1.0.1]: https://pypi.org/project/rnakinet/1.0.1/
[1.0.0]: https://pypi.org/project/rnakinet/1.0.0/
[0.1.26]: https://pypi.org/project/rnakinet/0.1.26/
[0.1.29]: https://pypi.org/project/rnakinet/0.1.29/
[0.1.24]: https://pypi.org/project/rnakinet/0.1.24/
[0.1.23]: https://pypi.org/project/rnakinet/0.1.23/
[0.1.20]: https://pypi.org/project/rnakinet/0.1.20/
[0.1.22]: https://pypi.org/project/rnakinet/0.1.22/
[0.1.19]: https://pypi.org/project/rnakinet/0.1.19/
[0.1.17]: https://pypi.org/project/rnakinet/0.1.17/
[0.1.18]: https://pypi.org/project/rnakinet/0.1.18/
[0.1.0]: https://pypi.org/project/rnakinet/0.1.0/
[0.1.16]: https://pypi.org/project/rnakinet/0.1.16/
