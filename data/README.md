# Data availability

The GitHub repository contains only a small, runnable sample because the complete SNAP data tree is approximately 25.8 GB and contains about 119,678 files.

## Bundled sample

The sample preserves the paths expected by the existing loaders:

```text
data/
├── epanet/
├── train_data/BA/20_nodes/0*.npy
└── test_data/BA/20_nodes/0*.npy
```

It contains one BA 20-node training topology and one BA 20-node test topology, together with their graph-distance, delayed-distance, and source-delay arrays. It is intended for loader checks and small functional tests, not for reproducing the reported full-scale metrics.

## Complete dataset

The complete dataset will be published separately because it is too large for the GitHub source repository.

- Dataset repository: **TODO: add the public dataset URL**
- Dataset version: **TODO**
- Archive checksums: **TODO**
- License: **TODO: select and add a data license**

After downloading, place the complete files under `data/train_data` and `data/test_data` without changing the directory hierarchy described in `data.md`.

For a machine-learning-oriented release, a public Hugging Face Dataset repository is recommended. For a fixed scholarly snapshot, additionally archive the released version on Zenodo and record its DOI here.
