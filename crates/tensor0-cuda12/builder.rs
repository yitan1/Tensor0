use std::{collections::hash_map::DefaultHasher, env, fs, hash::Hash, path::{Path, PathBuf}, process::Command, time::Instant};
mod build_helpers;
use build_helpers::{compiler_path, dependencies, fingerprint};
fn compile_cuda(
    manifest_dir: &Path, output_dir: &Path, include_dir: &Path, opt_level: &str,
    host_compiler: &Path, host_identity: &DefaultHasher,
) {
    let nvcc = env::var_os("NVCC").unwrap_or_else(|| {
        env::var_os("CUDA_HOME").map(|root| PathBuf::from(root).join("bin/nvcc").into_os_string())
            .unwrap_or_else(|| "nvcc".into())
    });
    let selected = compiler_path(&nvcc);
    let resolved = fs::canonicalize(&selected).expect("resolve NVCC");
    let cuda_home = env::var_os("CUDA_HOME").map(PathBuf::from)
        .unwrap_or_else(|| resolved.parent().unwrap().parent().unwrap().to_path_buf());
    let version = Command::new(&selected).arg("--version").output().expect("failed to identify NVCC");
    assert!(version.status.success(), "failed to identify NVCC");
    let mut objects = Vec::new();
    for source in ["native/cuda/copy.cu", "native/cuda/update.cu", "native/cuda/accumulation.cu", "native/cuda/dot.cu", "native/cuda/reduction.cu"] {
        let object = output_dir.join(source.replace('/', "_").replace(".cu", ".o"));
        let depfile = object.with_extension("d");
        let stamp = object.with_extension("fingerprint");
        let arch = env::var("TENSOR0_CUDA_ARCH").unwrap_or_else(|_| "sm_80".into());
        let mut command = Command::new(&selected);
        command.env_remove("NVCC_CCBIN")
            .arg("-ccbin").arg(host_compiler)
            .args(["-std=c++20", "-DNDEBUG", "-Xcompiler", "-fPIC", "-Xcompiler", "-fvisibility=default"])
            // NVCC has no size-optimization level; keep its optimization explicit.
            .arg(format!("-O{}", if matches!(opt_level, "s" | "z") { "2" } else { opt_level }))
            .arg(format!("-arch={arch}"))
            .arg("-I").arg(include_dir)
            .args(["-MD", "-MT", "tensor0-object", "-MF"]).arg(&depfile)
            .arg("-c").arg(source).arg("-o").arg(&object).current_dir(manifest_dir);
        let mut inputs = host_identity.clone();
        include_bytes!("builder.rs").hash(&mut inputs);
        format!("{command:?}").hash(&mut inputs);
        version.stdout.hash(&mut inputs);
        version.stderr.hash(&mut inputs);
        for name in ["NVCC", "CUDA_HOME", "TENSOR0_CUDA_ARCH", "PATH", "CPATH", "CPLUS_INCLUDE_PATH",
                     "COMPILER_PATH", "GCC_EXEC_PREFIX", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN"] {
            println!("cargo:rerun-if-env-changed={name}");
            env::var_os(name).hash(&mut inputs);
        }
        for path in [&selected, &resolved] {
            println!("cargo:rerun-if-changed={}", path.display());
            path.hash(&mut inputs);
            fs::read(path).expect("read NVCC").hash(&mut inputs);
        }
        let previous = dependencies(&depfile, manifest_dir);
        let expected = previous.as_ref().and_then(|paths| fingerprint(&inputs, paths));
        if !object.is_file() || expected.is_none() || fs::read_to_string(&stamp).ok() != expected {
            fs::write(&stamp, "").expect("invalidate CUDA fingerprint");
            eprintln!("tensor0-native: compiling {source}");
            let started = Instant::now();
            let status = command.status().expect("failed to launch NVCC; tensor0-cuda12 requires a CUDA toolkit");
            eprintln!("tensor0-native: {} {source} in {:.2}s",
                if status.success() { "compiled" } else { "failed" }, started.elapsed().as_secs_f64());
            assert!(status.success(), "failed to compile Tensor0 CUDA stride FFI");
            assert!(object.is_file(), "NVCC did not produce CUDA object");
            let paths = dependencies(&depfile, manifest_dir).expect("read NVCC dependency file");
            fs::write(&stamp, fingerprint(&inputs, &paths).expect("read CUDA dependencies"))
                .expect("save CUDA fingerprint");
        } else {
            eprintln!("tensor0-native: cached {source}");
        }
        for path in dependencies(&depfile, manifest_dir).expect("read NVCC dependency file") {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        objects.push(object);
    }
    // These independent PIC units provide the layout helpers used by CUDA.
    objects.extend(["native_layout_descriptor.o", "native_layout_record.o"]
        .map(|name| output_dir.join(name)));
    let forwarder = output_dir.join("cudart_forwarder.o");
    let depfile = forwarder.with_extension("d");
    let forwarder_stamp = forwarder.with_extension("fingerprint");
    let mut compile = Command::new(host_compiler);
    compile.args(["-std=c++20", "-O2", "-fPIC", "-fvisibility=default", "-pthread"])
        .arg("-I").arg(cuda_home.join("include"))
        .args(["-MD", "-MT", "tensor0-object", "-MF"]).arg(&depfile)
        .arg("-c").arg("native/cudart_forwarder.cc").arg("-o").arg(&forwarder)
        .current_dir(manifest_dir);
    let mut seed = host_identity.clone();
    include_bytes!("builder.rs").hash(&mut seed);
    format!("{compile:?}").hash(&mut seed);
    let expected = dependencies(&depfile, manifest_dir).and_then(|paths| fingerprint(&seed, &paths));
    if !forwarder.is_file() || expected.is_none() || fs::read_to_string(&forwarder_stamp).ok() != expected {
        fs::write(&forwarder_stamp, "").unwrap();
        assert!(compile.status().expect("compile CUDA forwarder").success(), "failed to compile CUDA forwarder");
        let paths = dependencies(&depfile, manifest_dir).expect("CUDA forwarder dependencies");
        fs::write(&forwarder_stamp, fingerprint(&seed, &paths).unwrap()).unwrap();
    }
    for path in dependencies(&depfile, manifest_dir).expect("CUDA forwarder dependencies") {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    objects.push(forwarder);
    let exports = manifest_dir.join("native/cudart_forwarder.map");
    println!("cargo:rerun-if-changed={}", exports.display());
    let companion = output_dir.join("_stride_cuda.so");
    let stamp = companion.with_extension("fingerprint");
    let mut command = Command::new(host_compiler);
    command.env_remove("LD_RUN_PATH")
        .args(["-shared", "-Wl,-z,defs"]).args(&objects)
        .arg(format!("-Wl,--version-script={}", exports.display()))
        .args(["-ldl", "-pthread"])
        .arg("-o").arg(&companion);
    let mut inputs = host_identity.clone();
    include_bytes!("builder.rs").hash(&mut inputs);
    format!("{command:?}").hash(&mut inputs);
    let mut paths = objects;
    paths.push(exports);
    let expected = fingerprint(&inputs, &paths).expect("read CUDA companion link inputs");
    if !companion.is_file() || fs::read_to_string(&stamp).ok().as_ref() != Some(&expected) {
        fs::write(&stamp, "").expect("invalidate CUDA companion fingerprint");
        let status = command.status().expect("failed to link CUDA companion");
        assert!(status.success(), "failed to link Tensor0 CUDA companion");
        assert!(companion.is_file(), "linker did not produce CUDA companion");
        fs::write(&stamp, expected).expect("save CUDA companion fingerprint");
    }
}

fn main() {
    let mut args = env::args_os().skip(1);
    let manifest_dir = fs::canonicalize(args.next().expect("source directory")).unwrap();
    let output_dir = PathBuf::from(args.next().expect("output directory"));
    fs::create_dir_all(&output_dir).expect("create output directory");
    let output_dir = fs::canonicalize(output_dir).unwrap();
    let include_dir = manifest_dir.join("vendor/jaxlib-0.10.1/include");
    let compiler = compiler_path(&env::var_os("CXX").unwrap_or_else(|| "/usr/bin/c++".into()));
    let resolved = fs::canonicalize(&compiler).expect("resolve C++ compiler");
    let version = Command::new(&compiler).arg("--version").output().unwrap();
    assert!(version.status.success(), "identify C++ compiler");
    let opt = env::var("TENSOR0_CXX_OPT_LEVEL").unwrap_or_else(|_| "3".into());
    assert!(matches!(opt.as_str(), "0" | "1" | "2" | "3" | "s" | "z"), "C++ optimization level must be 0, 1, 2, 3, s, or z");
    let mut identity = DefaultHasher::new();
    compiler.hash(&mut identity);
    resolved.hash(&mut identity);
    fs::read(&resolved).unwrap().hash(&mut identity);
    version.stdout.hash(&mut identity);
    version.stderr.hash(&mut identity);
    include_bytes!("builder.rs").hash(&mut identity);
    include_bytes!("build_helpers.rs").hash(&mut identity);
    for name in ["PATH", "CPATH", "CPLUS_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX"] {
        env::var_os(name).hash(&mut identity);
    }
    for source in ["native/layout/descriptor.cc", "native/layout/record.cc"] {
        let object = output_dir.join(source.replace('/', "_").replace(".cc", ".o"));
        let depfile = object.with_extension("d");
        let stamp = object.with_extension("fingerprint");
        let mut command = Command::new(&compiler);
        command.args(["-std=c++20", "-DNDEBUG", "-fPIC", "-Wall", "-Wextra", "-Wpedantic"])
            .arg(format!("-O{}", if opt == "z" { "s" } else { &opt }))
            .args(["-MD", "-MT", "tensor0-object", "-MF"]).arg(&depfile)
            .arg("-c").arg(source).arg("-o").arg(&object).current_dir(&manifest_dir);
        let mut seed = identity.clone();
        format!("{command:?}").hash(&mut seed);
        let expected = dependencies(&depfile, &manifest_dir).and_then(|paths| fingerprint(&seed, &paths));
        if !object.is_file() || expected.is_none() || fs::read_to_string(&stamp).ok() != expected {
            fs::write(&stamp, "").unwrap();
            assert!(command.status().expect("compile layout").success(), "failed to compile {source}");
            assert!(object.is_file(), "compiler did not produce {source} object");
            let paths = dependencies(&depfile, &manifest_dir).expect("layout dependencies");
            fs::write(&stamp, fingerprint(&seed, &paths).unwrap()).unwrap();
        }
    }
    compile_cuda(&manifest_dir, &output_dir, &include_dir, &opt, &compiler, &identity);
}
