# Tensor0

Tensor0 is an experimental symmetric tensor library for Python, with Rust-backed
structural metadata and JAX-backed storage and automatic differentiation. It is
not yet a production-ready tensor-network framework. Before 1.0, public API
patch releases remain backward-compatible; breaking changes require a minor
release and migration notes.

Use it for tensor algebra with conserved charges and other supported symmetries:
work with symmetry-allowed blocks, then compose, contract or differentiate them
using JAX. It provides tensor primitives rather than a complete simulation
algorithm framework.

**[Documentation](https://yitan1.github.io/Tensor0/)** ·
[Installation](docs/installation.md) · [Usage](docs/usage.md) ·
[API](docs/api.md) · [Support policy](docs/support.md)

## Installation

**[Tensor0 0.1.0](https://pypi.org/project/tensor0/0.1.0/) was published on
2026-10-10.** Tested CPU wheels are available for **CPython 3.11–3.14 on
Linux x86_64 with glibc >=2.28**. Install in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install tensor0==0.1.0
```

These wheels require **no Rust or C++ compiler**. JAX and JAXLIB remain pinned to
**0.10.1**; the native XLA FFI checks these versions exactly. CPU installation
does not require CUDA.

For source builds, **Rust >=1.87** and a **C++20-capable compiler** are required;
see [source installation and memory limits](docs/installation.md#source-install).
[Optional CUDA support](docs/cuda.md) is experimental and restricted to selected
operations, dtypes and AD paths—not the full TensorMap API or full reverse AD.

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

Tensor0 provides graded spaces and built-in sector families, ordinary and
diagonal tensor maps, blockwise linear algebra and truncation, transforms and
symmetry-aware contractions, and JAX `jit`/`grad` integration. See the
[usage guide](docs/usage.md), [contraction guide](docs/contractions.md), and
[executable examples](docs/examples.md) for capabilities and boundaries.

## Contributing and benchmarks

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and verification links,
[CHANGELOG.md](CHANGELOG.md) for release changes, and the
[release checklist](docs/releasing.md) for installed-artifact acceptance.
Reproducible [benchmark suites](benchmarks/README.md) and their reports document
measured configurations and limitations; they are not performance guarantees.

## Acknowledgments

Tensor0's main design and architecture are based on
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl), bringing these ideas to
Python with JAX-backed storage and automatic differentiation.

For citation metadata, see [CITATION.cff](CITATION.cff).
