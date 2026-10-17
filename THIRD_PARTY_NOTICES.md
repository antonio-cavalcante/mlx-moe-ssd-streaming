# Third-party notices

## Code

This repository contains no third-party source code. It imports and calls mlx-lm (below), including its
`muse_spark` model, which it does not copy.

## Dependencies (not included in this repository)

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) - MIT, Apple Inc.
- NumPy, huggingface_hub - installed from PyPI under their own licenses.

## The model (not included in this repository)

This repository contains **no model weights and no copy of the model's files**. The
model it is built to run, `mlx-community/Muse-Spark-Open-816B-A42B-oQ3e` (a quantization
of Meta's Muse Spark Open), is downloaded separately by the user and is governed by the
**Muse Model License** and its Usage Policy, which are shipped in the model download
(`LICENSE.md`, `USE_POLICY.md`). Use of the model is subject to those terms, including the
requirement to display "Built with Muse Spark Open" and to comply with the Usage Policy.

This project is independent. It is not affiliated with, endorsed by, or sponsored by
Meta, Apple, or the mlx-community.
