use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

const VENDORED_JAX_VERSION: &str = "0.10.1";
const VENDORED_JAXLIB_VERSION: &str = "0.10.1";
const FFI_HEADERS: [&str; 3] = ["api.h", "c_api.h", "ffi.h"];
const STRIDE_NATIVE_SOURCES: [&str; 9] = [
    "native/execute/scheduling.cc",
    "native/ffi/prepared.cc",
    "native/layout/blocking.cc",
    "native/layout/record.cc",
    "native/layout/traversal.cc",
    "native/ffi/copy.cc",
    "native/ffi/update.cc",
    "native/ffi/reduction.cc",
    "native/ffi/dot.cc",
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


// -MT fixes the target name, so only the dependency side needs Make unescaping.
fn dependencies(path: &Path, manifest_dir: &Path) -> Option<Vec<PathBuf>> {
    let text = fs::read_to_string(path).ok()?.replace("\\\n", "");
    let (_, words) = text.split_once(':')?;
    let mut paths = Vec::new();
    let mut word = String::new();
    let mut chars = words.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => word.push(chars.next()?),
            '$' if chars.peek() == Some(&'$') => { chars.next(); word.push('$'); }
            c if c.is_whitespace() => {
                if !word.is_empty() {
                    paths.push(manifest_dir.join(&word));
                    word.clear();
                }
            }
            c => word.push(c),
        }
    }
    if !word.is_empty() { paths.push(manifest_dir.join(word)); }
    if paths.is_empty() { None } else { Some(paths) }
}

fn fingerprint(seed: &DefaultHasher, paths: &[PathBuf]) -> Option<String> {
    let mut inputs = seed.clone();
    for path in paths {
        path.hash(&mut inputs);
        fs::read(path).ok()?.hash(&mut inputs);
    }
    Some(format!("{:016x}", inputs.finish()))
}

fn compiler_path(compiler: &std::ffi::OsStr) -> PathBuf {
    let path = Path::new(compiler);
    if path.components().count() > 1 {
        return env::current_dir().expect("current directory").join(path);
    }
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(path))
        .find(|candidate| candidate.is_file())
        .expect("C++ compiler was not found in PATH")
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(tensor0_stride_ffi)");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=TENSOR0_CXX_OPT_LEVEL");
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
    let opt_level = match env::var("TENSOR0_CXX_OPT_LEVEL") {
        Ok(level) => level,
        Err(env::VarError::NotPresent) => env::var("OPT_LEVEL").expect("OPT_LEVEL"),
        Err(error) => panic!("invalid TENSOR0_CXX_OPT_LEVEL: {error}"),
    };
    assert!(
        matches!(opt_level.as_str(), "0" | "1" | "2" | "3" | "s" | "z"),
        "C++ optimization level must be 0, 1, 2, 3, s, or z; got {opt_level:?}"
    );
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
    let selected_compiler = compiler_path(&compiler);
    let resolved_compiler = fs::canonicalize(&selected_compiler).expect("resolve C++ compiler");
    println!("cargo:rerun-if-changed={}", selected_compiler.display());
    println!("cargo:rerun-if-changed={}", resolved_compiler.display());
    let compiler_bytes = fs::read(&resolved_compiler).expect("read C++ compiler");
    for source in STRIDE_NATIVE_SOURCES {
        let object = output_dir.join(source.replace('/', "_").replace(".cc", ".o"));
        let depfile = object.with_extension("d");
        let mut command = Command::new(&selected_compiler);
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
            .args(["-MD", "-MT", "tensor0-object", "-MF"])
            .arg(&depfile)
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
        selected_compiler.hash(&mut inputs);
        resolved_compiler.hash(&mut inputs);
        compiler_bytes.hash(&mut inputs);
        let stamp = object.with_extension("fingerprint");
        let previous = dependencies(&depfile, &manifest_dir);
        let expected = previous.as_ref().and_then(|paths| fingerprint(&inputs, paths));
        if !object.is_file() || expected.is_none() || fs::read_to_string(&stamp).ok() != expected {
            fs::write(&stamp, "").expect("failed to invalidate stride FFI build fingerprint");
            let status = command.status()
                .expect("failed to launch the C++ compiler for Tensor0 stride FFI");
            assert!(status.success(), "failed to compile Tensor0 stride FFI: {source}");
            assert!(object.is_file(), "compiler did not produce {source} object");
            let paths = dependencies(&depfile, &manifest_dir).expect("read compiler dependency file");
            let fresh = fingerprint(&inputs, &paths).expect("read compiler dependencies");
            fs::write(&stamp, fresh).expect("failed to save stride FFI build fingerprint");
        }
        for dependency in dependencies(&depfile, &manifest_dir).expect("read compiler dependencies") {
            println!("cargo:rerun-if-changed={}", dependency.display());
        }
        println!("cargo:rustc-link-arg={}", object.display());
    }

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=tensor0_stride_ffi");
}
