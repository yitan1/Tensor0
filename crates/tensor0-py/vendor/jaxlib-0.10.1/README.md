# Vendored JAXLIB FFI headers

This directory contains an unmodified minimal header snapshot from the
`jaxlib==0.10.1` wheel installed for the Tensor0 stride compatibility point.

Upstream: <https://github.com/jax-ml/jax>

Package metadata:

- package: `jaxlib`
- version: `0.10.1`
- license: Apache-2.0
- source include root: `jaxlib/include`

Only the transitive quoted-include closure required by the native FFI handlers
in `crates/tensor0-py/native/ffi/` and `crates/tensor0-cuda12/native/cuda/`
is retained:

| File | SHA-256 |
| --- | --- |
| `include/xla/ffi/api/api.h` | `7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb` |
| `include/xla/ffi/api/c_api.h` | `85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539` |
| `include/xla/ffi/api/ffi.h` | `4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5` |
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
