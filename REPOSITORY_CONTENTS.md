# Public repository contents

This directory is a publication copy of SNAP. Files in the parent working directory were copied, not moved or deleted.

## Included

- Core training, evaluation, graph-generation, scenario, and plotting code.
- Regression tests and the Python dependency list.
- A minimal BA 20-node sample under `data/train_data` and `data/test_data`.
- EPANET network definitions under `data/epanet`.
- Final `ppo_mix_v3_sd` and `ppo_mix_v3_no_sd` model artifacts.
- Uncompressed experiment results under `result`.
- Figure outputs under `fig5_output`, `fig6_barlog_refined`, and `figure_save`.

## Intentionally excluded

- The complete data tree (approximately 25.8 GB and 119,678 files).
- Historical and intermediate model checkpoints (approximately 886 MB in the working tree).
- Duplicate result ZIP archives; the corresponding unpacked results are included.
- IDE configuration, caches, local logs, backup archives, exploratory scripts, and personal working notes.

## External publication still required

Before publishing the GitHub repository, publish the complete dataset and optional historical checkpoints to a dedicated dataset/model host. Then replace the placeholders in `data/README.md` and `model_save/README.md` with stable URLs and checksums.

No software or data license is assigned by this preparation step. Add licenses only after the copyright holder selects the intended terms.
