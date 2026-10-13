# Vendored JAXLIB FFI headers

This directory contains an unmodified minimal header snapshot from the
`jaxlib==0.11.2` wheel installed for the Tensor0 stride compatibility point.

Upstream: <https://github.com/jax-ml/jax>

Package metadata:

- package: `jaxlib`
- version: `0.11.2`
- license: Apache-2.0
- source include root: `jaxlib/include`

Only the transitive quoted-include closure required by the native FFI handlers
in `crates/tensor0-py/native/ffi/` and `crates/tensor0-cuda12/native/cuda/`
is retained:

| File | SHA-256 |
| --- | --- |
| `include/xla/ffi/api/api.h` | `1426ea8206a1b9f1824f460572ea731665eb7cb9def1f0fc166b13df4b955ad0` |
| `include/xla/ffi/api/c_api.h` | `9847993912aa1aadb2dac4af16eeb9d8bae3f470905b519977d1d09b3f37eedf` |
| `include/xla/ffi/api/ffi.h` | `0878473a1ee35fdd9ff15424d46a44602fb53019f0d0a9f5117abdb386c35129` |
| `LICENSE.txt` | `e3d8688a2c75d4e33641cc88046a8a1593ee79822411fa45a7bd32e37f847d29` |

`build.rs` embeds the exact build JAX/JAXLIB version identity. Python disables
native target registration unless the runtime JAX and JAXLIB versions match
that identity exactly.

Tensor0-owned code remains MIT-licensed. Both distributions incorporate these
Apache-2.0 FFI headers and declare `MIT AND Apache-2.0` in their license metadata.
Both wheels carry the canonical, unmodified `LICENSE.txt` under
`.dist-info/licenses/` alongside Tensor0's `LICENSE`. The upstream license file
also contains notices for other jaxlib components; shipping it intact does not
mean those components are included in Tensor0. No separate NOTICE file is
present in this vendored snapshot; the headers retain their upstream copyright
and license notices. This metadata does not cover separately supplied CUDA
toolchain, runtime or driver components.
