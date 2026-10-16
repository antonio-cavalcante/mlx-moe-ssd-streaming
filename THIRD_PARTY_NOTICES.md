# Third-party notices

## `src/muse_spark_1975.py` - mlx-lm (MIT)

This file is the Muse Spark model implementation from the (closed, unmerged)
mlx-lm pull request #1975 (`jundot/mlx-lm@117b84c`, `mlx_lm/models/muse_spark.py`),
with only its relative imports changed to absolute `mlx_lm.models.*` imports.
It is part of mlx-lm, which is distributed under the following license:

```
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies (not included in this repository)

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) - MIT, Apple Inc.
- NumPy, huggingface_hub - installed from PyPI under their own licenses.

## The model (not included in this repository)

This repository contains **no model weights and no copy of the model's files**. The
model it is built to run, `mlx-community/Muse-Spark-Open-816B-A42B-3bit` (a quantization
of Meta's Muse Spark Open), is downloaded separately by the user and is governed by the
**Muse Model License** and its Usage Policy, which are shipped in the model download
(`LICENSE.md`, `USE_POLICY.md`). Use of the model is subject to those terms, including the
requirement to display "Built with Muse Spark Open" and to comply with the Usage Policy.

This project is independent. It is not affiliated with, endorsed by, or sponsored by
Meta, Apple, or the mlx-community.
