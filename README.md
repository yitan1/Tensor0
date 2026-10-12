# Tensor0

**Symmetry-aware tensors for Python, powered by Rust and JAX.**

[![PyPI version](https://img.shields.io/pypi/v/tensor0)](https://pypi.org/project/tensor0/)
[![CI](https://github.com/yitan1/Tensor0/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/yitan1/Tensor0/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://yitan1.github.io/Tensor0/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Work with conserved charges and symmetry-allowed tensor blocks, then compose,
contract or differentiate them using JAX. Rust handles structural metadata;
JAX handles numerical storage and automatic differentiation.

> **Pre-1.0 software.** [CUDA support](docs/cuda.md) is experimental.

**[Documentation](https://yitan1.github.io/Tensor0/)** ·
[Installation](docs/installation.md) · [Usage](docs/usage.md) ·
[API](docs/api.md) · [Support policy](docs/support.md)

## Installation

Install from PyPI in a virtual environment:

```bash
# CPU
python -m pip install tensor0

# CUDA 12 (experimental)
python -m pip install "tensor0[cuda12]"
```

See [CUDA support](docs/cuda.md) for GPU requirements and supported operations.
Prebuilt wheels require no Rust or C++ compiler. See the
[installation guide](docs/installation.md) for available platforms, supported
Python versions, GPU setup and source builds.

## A small symmetric tensor

```python
import jax.numpy as jnp
from tensor0 import TensorMap, U1Irrep, hom, space, storage_dim

v = space(U1Irrep, {0: 2, 1: 3})
h = hom((v,), (v,))
tensor = TensorMap(h, jnp.arange(storage_dim(h), dtype=jnp.float32))

for sector, block in tensor.blocks():
    print(sector, block.shape)
```

## Main capabilities

- **Symmetry-aware tensors** — graded spaces, built-in sector families, and
  ordinary and diagonal tensor maps with symmetry-allowed blocks.
- **Tensor algebra** — transforms, contractions, blockwise linear algebra,
  decompositions and truncation.
- **JAX integration** — `jit` and automatic differentiation within documented
  operation-specific limits.

Start with the [usage guide](docs/usage.md), [contraction guide](docs/contractions.md)
and [executable examples](docs/examples.md). Before 1.0, documented public API
patch releases remain backward-compatible; breaking changes require a minor
release and migration notes. See the [support policy](docs/support.md).

## Contributing and benchmarks

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and verification links,
[CHANGELOG.md](CHANGELOG.md) for release changes, and the
[release checklist](docs/releasing.md) for installed-artifact acceptance.
See the reproducible [benchmark suites](benchmarks/README.md) for measurements
and tested configurations.

## Acknowledgments

Tensor0's main design and architecture are based on
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl), bringing these ideas to
Python with JAX-backed storage and automatic differentiation.

For citation metadata, see [CITATION.cff](CITATION.cff).
