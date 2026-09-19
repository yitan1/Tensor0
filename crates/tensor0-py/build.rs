use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

const VENDORED_JAX_VERSION: &str = "0.10.1";
const VENDORED_JAXLIB_VERSION: &str = "0.10.1";
const FFI_HEADERS: [&str; 3] = ["api.h", "c_api.h", "ffi.h"];
const STRIDE_NATIVE_SOURCES: [&str; 17] = [
    "native/stride_ffi.cc",
    "native/numeric/scalar.inc",
    "native/numeric/expression.inc",
    "native/layout/record.inc",
    "native/layout/traversal.inc",
    "native/layout/blocking.inc",
    "native/ffi/dtype.inc",
    "native/ffi/descriptor.inc",
    "native/execute/scheduling.inc",
    "native/kernels/generic.inc",
    "native/kernels/specialized.inc",
    "native/kernels/dispatch.inc",
    "native/kernels/avx2.inc",
    "native/execute/map.inc",
    "native/ffi/handlers.inc",
    "native/ffi/bindings.inc",
    "native/execute/reduction.inc",
];

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
    for source in STRIDE_NATIVE_SOURCES {
        println!("cargo:rerun-if-changed={source}");
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        println!("cargo:warning=Tensor0 stride FFI is disabled on unsupported target {target_os}");
        return;
    }

    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let Some(build_info) = jax_build_info(&manifest_dir) else {
        println!(
            "cargo:warning=vendored JAXLIB {VENDORED_JAXLIB_VERSION} FFI headers were not found; building Tensor0 without native CPU stride execution"
        );
        return;
    };
    let output_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let compiler = env::var_os("CXX").unwrap_or_else(|| "/usr/bin/c++".into());
    let compiler_version = Command::new(&compiler)
        .arg("--version")
        .output()
        .expect("failed to identify the C++ compiler for Tensor0 stride FFI");
    assert!(compiler_version.status.success(), "failed to identify the C++ compiler");
    let opt_level = env::var("OPT_LEVEL").expect("OPT_LEVEL");
    let optimization = format!("-O{}", if opt_level == "z" { "s" } else { &opt_level });
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
    let source = "native/stride_ffi.cc";
    let object = output_dir.join("tensor0_stride_ffi.o");
    let mut command = Command::new(&compiler);
    command
        .args([
            "-std=c++20",
            &optimization,
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
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(&object)
        .current_dir(&manifest_dir);
    let mut inputs = DefaultHasher::new();
    include_bytes!("build.rs").hash(&mut inputs);
    format!("{command:?}").hash(&mut inputs);
    compiler_version.stdout.hash(&mut inputs);
    compiler_version.stderr.hash(&mut inputs);
    for name in ["PATH", "CPATH", "CPLUS_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX"] {
        println!("cargo:rerun-if-env-changed={name}");
        env::var_os(name).hash(&mut inputs);
    }
    for dependency in STRIDE_NATIVE_SOURCES
        .iter()
        .map(|dependency| manifest_dir.join(dependency))
        .chain(FFI_HEADERS.iter().map(|header| build_info.include_dir.join("xla/ffi/api").join(header)))
    {
        dependency.hash(&mut inputs);
        fs::read(&dependency).expect("failed to read stride FFI build input").hash(&mut inputs);
    }
    let fingerprint = format!("{:016x}", inputs.finish());
    let stamp = object.with_extension("fingerprint");
    if !object.is_file() || fs::read_to_string(&stamp).ok().as_deref() != Some(&fingerprint) {
        fs::write(&stamp, "").expect("failed to invalidate stride FFI build fingerprint");
        let status = command.status()
            .expect("failed to launch the C++ compiler for Tensor0 stride FFI");
        if !status.success() {
            panic!("failed to compile Tensor0 stride FFI: {source}");
        }
        fs::write(&stamp, &fingerprint).expect("failed to save stride FFI build fingerprint");
    }

    println!("cargo:rustc-link-arg={}", object.display());
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=tensor0_stride_ffi");
}
