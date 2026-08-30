use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const VENDORED_JAX_VERSION: &str = "0.10.1";
const VENDORED_JAXLIB_VERSION: &str = "0.10.1";
const FFI_HEADERS: [&str; 3] = ["api.h", "c_api.h", "ffi.h"];

struct JaxBuildInfo {
    include_dir: PathBuf,
    jax_version: &'static str,
    jaxlib_version: &'static str,
}

fn jax_build_info(manifest_dir: &Path) -> Option<JaxBuildInfo> {
    let include_dir = manifest_dir
        .join("vendor")
        .join(format!("jaxlib-{VENDORED_JAXLIB_VERSION}"))
        .join("include");
    if FFI_HEADERS
        .iter()
        .all(|header| include_dir.join("xla/ffi/api").join(header).is_file())
    {
        return Some(JaxBuildInfo {
            include_dir,
            jax_version: VENDORED_JAX_VERSION,
            jaxlib_version: VENDORED_JAXLIB_VERSION,
        });
    }
    None
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(tensor0_stride_ffi)");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-changed=native/stride_descriptor.h");
    println!("cargo:rerun-if-changed=native/stride_ffi.cc");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        println!("cargo:warning=Tensor0 stride FFI is disabled on unsupported target {target_os}");
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let Some(build_info) = jax_build_info(&manifest_dir) else {
        println!(
            "cargo:warning=vendored JAXLIB {VENDORED_JAXLIB_VERSION} FFI headers were not found; building Tensor0 with stride fallback only"
        );
        return;
    };
    let output_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let object = output_dir.join("tensor0_stride_ffi.o");
    let compiler = env::var_os("CXX").unwrap_or_else(|| "/usr/bin/c++".into());
    for header in FFI_HEADERS {
        println!(
            "cargo:rerun-if-changed={}",
            build_info
                .include_dir
                .join("xla/ffi/api")
                .join(header)
                .display()
        );
    }
    let status = Command::new(&compiler)
        .args([
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        ])
        .arg("-isystem")
        .arg(&build_info.include_dir)
        .arg(format!(
            "-DTENSOR0_STRIDE_JAX_VERSION=\"{}\"",
            build_info.jax_version
        ))
        .arg(format!(
            "-DTENSOR0_STRIDE_JAXLIB_VERSION=\"{}\"",
            build_info.jaxlib_version
        ))
        .arg("-Inative")
        .arg("-c")
        .arg("native/stride_ffi.cc")
        .arg("-o")
        .arg(&object)
        .current_dir(&manifest_dir)
        .status()
        .expect("failed to launch the C++ compiler for Tensor0 stride FFI");
    if !status.success() {
        panic!("failed to compile Tensor0 stride FFI");
    }

    println!("cargo:rustc-link-arg={}", object.display());
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=tensor0_stride_ffi");
}
