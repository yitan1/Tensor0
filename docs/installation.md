# Installation

## Install from PyPI

The current PyPI release is
[Tensor0 0.2.0](https://pypi.org/project/tensor0/0.2.0/), published on 2026-10-12.

### Published 0.2.0 platform availability

| Platform | Prebuilt core wheels | Status |
| --- | --- | --- |
| Linux x86_64, glibc >=2.28 | CPython 3.11–3.14, manylinux_2_28 | Available and tested |
| macOS 15+ ARM64 | CPython 3.11–3.14, macosx_15_0_arm64 | Available and tested on hosted macOS 15 |
| Windows | Not provided | Outside the current support scope |

All eight core wheels were built from one shared sdist and passed installed-wheel
acceptance. macOS is CPU-only: do not install the Linux-only `cuda12` extra there.
The macOS wheels are ARM64, not Intel or universal2, and require macOS 15 or newer.
See [Releasing](releasing.md) for the artifact acceptance policy.

Other platforms have not been validated. Lack of a wheel is not proof that
source builds are impossible; neither is it a guarantee that they work.

Use a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install tensor0==0.2.0
python -m pip check
```

Installing these prebuilt wheels requires **no Rust or C++ compiler** and no CUDA
toolkit. JAX and JAXLIB **0.10.1** are installed through Tensor0's pinned
dependencies. Native handlers use matching vendored XLA FFI headers and reject
mismatched runtime versions. Do not independently upgrade JAX/JAXLIB.

Python >=3.11 package metadata does not establish support for future interpreters
or other platforms. Documentation on `main` describes the current checkout;
consult the matching release documentation when using an older artifact.

## Source build prerequisites

These tools are needed only when building from source, not when installing a
supported wheel:

- Rust/Cargo **1.87 or newer**, and a linker and standard system build tools
  (including `ar`).
- A **C++20-capable compiler** and its standard library. The build defaults to
  `/usr/bin/c++`; set `CXX` to select another compiler. `CXXFLAGS` is not used.
- A supported Python interpreter and access to the pinned Python dependencies
  and Cargo crates, or suitable local caches. The source-build procedure below
  has been validated on Linux x86_64 and macOS ARM64.

Native C++ template compilation is memory-intensive: heavy units can each use
several GiB. Cargo's default parallelism follows CPU count, not free memory or
container limits. Start with `CARGO_BUILD_JOBS=1` on constrained machines and
increase only within the available memory budget. Even a serial build may exceed
a small container's limit; there is no universal minimum-RAM or successful-build
guarantee. A killed compiler can indicate an out-of-memory limit—inspect system
or container logs before retrying. Compiler jobs are unrelated to runtime worker
threads. See [Development](development.md) for profiles, optimization overrides,
cache behavior and jobserver details.

## Source install

Use a virtual environment with the baseline interpreter:

```bash
git clone https://github.com/yitan1/Tensor0.git
cd Tensor0
git rev-parse HEAD
python3.11 -m venv .venv
source .venv/bin/activate
CARGO_BUILD_JOBS=1 python -m pip install .
python -m pip check
```

This is a non-editable source build. It requires access to Python build/runtime
dependencies and Cargo crates, or suitable local caches. No CUDA toolkit is
needed for the default CPU-only extension. For editable contributor setup,
follow [Local Setup](development.md#local-setup) instead.

## Check the installed package

Run outside the checkout, with the virtual environment still active, to avoid
accidentally testing source-tree imports:

```bash
cd /tmp
JAX_PLATFORMS=cpu python - <<'PY'
from importlib.metadata import version
import jax
import tensor0
from tensor0 import U1Irrep, identity, permute, space

print("Tensor0:", version("tensor0"), tensor0.__file__)
print("JAX/JAXLIB:", version("jax"), version("jaxlib"))
print("Devices:", jax.devices())
v = space(U1Irrep, {0: 2, 1: 1})
t = identity(v)
r = jax.jit(lambda x: permute(x, ((1,), (0,))))(t)
r.storage.data.block_until_ready()
print("Installed CPU smoke OK:", r.storage.data.shape)
PY
```

This is a smoke check, not comprehensive numerical acceptance. See the
[release checklist](releasing.md) for artifact validation and the
[support policy](support.md) for execution and API boundaries.

## Experimental CUDA

Version 0.2.0 provides a separate
[`tensor0-cuda12` companion](https://pypi.org/project/tensor0-cuda12/0.2.0/)
with an exact matching core version. Install the published pair using:

```bash
python -m pip install "tensor0[cuda12]==0.2.0"
python -m pip check
```

CUDA support currently targets Linux x86_64 and remains experimental.
Installing GPU-enabled JAX alone does not supply the Tensor0 companion.
Prebuilt companion wheels do not require a local CUDA toolkit; compatible
NVIDIA hardware, a driver and the specified runtime dependencies are needed
for execution. Building the companion from source additionally requires the
CUDA toolkit and host compiler described in the
[development guide](development.md#optional-cuda-stride-build).
See [CUDA support](cuda.md) for the tested operations, dtypes and AD limits.
