use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

const VENDORED_JAX_VERSION: &str = "0.10.1";
const VENDORED_JAXLIB_VERSION: &str = "0.10.1";
const FFI_HEADERS: [&str; 3] = ["api.h", "c_api.h", "ffi.h"];
const STRIDE_NATIVE_SOURCES: [&str; 12] = [
    "native/execute/scheduling.cc",
    "native/ffi/prepared.cc",
    "native/layout/blocking.cc",
    "native/layout/record.cc",
    "native/layout/traversal.cc",
    "native/ffi/copy.cc",
    "native/ffi/update_a.cc",
    "native/ffi/update_b.cc",
    "native/ffi/reduction_a.cc",
    "native/ffi/reduction_b.cc",
    "native/ffi/dot_a.cc",
    "native/ffi/dot_b.cc",
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
    // SAFETY: Read Cargo's inherited jobserver before starting any threads.
    let jobserver = unsafe { jobserver::Client::from_env() };
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
    // NUM_JOBS bounds demand; the jobserver grants actual extra CPU slots.
    let jobs = env::var("NUM_JOBS")
        .map(|value| value.parse::<usize>().expect("NUM_JOBS must be a positive integer"))
        .unwrap_or(1);
    assert!(jobs > 0, "NUM_JOBS must be a positive integer");
    let compile = |source: &str| {
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
            eprintln!("tensor0-native: compiling {source}");
            let started = Instant::now();
            let status = command.status()
                .expect("failed to launch the C++ compiler for Tensor0 stride FFI");
            eprintln!("tensor0-native: {} {source} in {:.2}s",
                if status.success() { "compiled" } else { "failed" },
                started.elapsed().as_secs_f64());
            assert!(status.success(), "failed to compile Tensor0 stride FFI: {source}");
            assert!(object.is_file(), "compiler did not produce {source} object");
            let paths = dependencies(&depfile, &manifest_dir).expect("read compiler dependency file");
            let fresh = fingerprint(&inputs, &paths).expect("read compiler dependencies");
            fs::write(&stamp, fresh).expect("failed to save stride FFI build fingerprint");
        } else {
            eprintln!("tensor0-native: cached {source}");
        }
        for dependency in dependencies(&depfile, &manifest_dir).expect("read compiler dependencies") {
            println!("cargo:rerun-if-changed={}", dependency.display());
        }
    };
    let jobs = if jobserver.is_some() { jobs.min(STRIDE_NATIVE_SOURCES.len()) } else { 1 };
    eprintln!("tensor0-native: checking {} units with up to {jobs} workers",
        STRIDE_NATIVE_SOURCES.len());
    let started = Instant::now();
    enum Event {
        Ready(std::io::Result<Option<jobserver::Acquired>>),
        Finished(Option<jobserver::Acquired>, thread::Result<()>),
    }
    let (sender, receiver) = mpsc::channel();
    // None represents the implicit slot Cargo already assigned to this script.
    sender.send(Event::Ready(Ok(None))).unwrap();
    let helper = jobserver.filter(|_| jobs > 1).map(|client| {
        let sender = sender.clone();
        let helper = client.into_helper_thread(move |token| {
            let _ = sender.send(Event::Ready(token.map(Some)));
        }).expect("failed to start native build jobserver helper");
        for _ in 1..jobs { helper.request_token(); }
        helper
    });
    let failure = thread::scope(|scope| {
        let mut next = 0;
        let mut running = 0;
        let mut failure = None;
        while running != 0 || (failure.is_none() && next < STRIDE_NATIVE_SOURCES.len()) {
            let slot = match receiver.recv().expect("native build event channel closed") {
                Event::Ready(token) => token.expect("failed to acquire native build job slot"),
                Event::Finished(slot, result) => {
                    running -= 1;
                    if let Err(error) = result {
                        if failure.is_none() { failure = Some(error); }
                    }
                    slot
                }
            };
            if failure.is_some() || next == STRIDE_NATIVE_SOURCES.len() {
                drop(slot);
                continue;
            }
            let source = STRIDE_NATIVE_SOURCES[next];
            next += 1;
            running += 1;
            let sender = sender.clone();
            let compile = &compile;
            scope.spawn(move || {
                let result = std::panic::catch_unwind(|| compile(source));
                let _ = sender.send(Event::Finished(slot, result));
            });
        }
        // Cancel unfulfilled requests: other Cargo jobs may hold every extra slot.
        drop(helper);
        failure
    });
    drop(receiver); // Return any acquired tokens still queued by the helper.
    if let Some(error) = failure { std::panic::resume_unwind(error); }
    eprintln!("tensor0-native: native objects ready in {:.2}s",
        started.elapsed().as_secs_f64());
    // Emit linker inputs in their original order, only after every worker succeeds.
    for source in STRIDE_NATIVE_SOURCES {
        let object = output_dir.join(source.replace('/', "_").replace(".cc", ".o"));
        println!("cargo:rustc-link-arg={}", object.display());
    }

    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=tensor0_stride_ffi");
}
