# Tensor0

Tensor0 is an experimental symmetric tensor library for Python, with Rust-backed
structural metadata and JAX-backed storage and automatic differentiation. It is
not yet a production-ready tensor-network framework; the pre-1.0 API may change.

Use it for tensor algebra with conserved charges and other supported symmetries:
work with symmetry-allowed blocks, then compose, contract or differentiate them
using JAX. It provides tensor primitives rather than a complete simulation
algorithm framework.

**[Documentation](https://yitan1.github.io/Tensor0/)** ·
[Installation](docs/installation.md) · [Usage](docs/usage.md) ·
[API](docs/api.md) · [Support policy](docs/support.md)

## Install from source

Source installation is the current distribution route. Published PyPI releases
and prebuilt wheels are not currently offered as an installation path.

The supported numerical baseline is **Linux CPU, Python 3.11, JAX and JAXLIB
0.10.1**. Builds require **Rust 1.87 or newer** and a **C++20-capable compiler**.
The native XLA FFI checks the JAX/JAXLIB versions exactly. Other Python versions
are CI-tested only where explicitly recorded, not a blanket support guarantee.

```bash
git clone https://github.com/yitan1/Tensor0.git
cd Tensor0
python3.11 -m venv .venv
source .venv/bin/activate
CARGO_BUILD_JOBS=1 python -m pip install .
```

Native compilation can use several GiB per heavy C++ unit. Serial compilation
limits concurrency, not peak memory; see [build prerequisites and memory
limits](docs/installation.md). CPU builds do not require CUDA.
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
[CHANGELOG.md](CHANGELOG.md) for unreleased changes, and the
[release checklist](docs/releasing.md) for installed-artifact acceptance.
Reproducible [benchmark suites](benchmarks/README.md) and their reports document
measured configurations and limitations; they are not performance guarantees.

## Acknowledgments

Tensor0's main design and architecture are based on
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl), bringing these ideas to
Python with JAX-backed storage and automatic differentiation.

For citation metadata, see [CITATION.cff](CITATION.cff).
