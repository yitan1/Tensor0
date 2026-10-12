use std::{collections::hash_map::DefaultHasher, env, fs, hash::{Hash, Hasher}, path::{Path, PathBuf}};
// -MT fixes the target name, so only the dependency side needs Make unescaping.
pub fn dependencies(path: &Path, manifest_dir: &Path) -> Option<Vec<PathBuf>> {
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

pub fn fingerprint(seed: &DefaultHasher, paths: &[PathBuf]) -> Option<String> {
    let mut inputs = seed.clone();
    for path in paths {
        path.hash(&mut inputs);
        fs::read(path).ok()?.hash(&mut inputs);
    }
    Some(format!("{:016x}", inputs.finish()))
}

pub fn compiler_path(compiler: &std::ffi::OsStr) -> PathBuf {
    let path = Path::new(compiler);
    if path.components().count() > 1 {
        return env::current_dir().expect("current directory").join(path);
    }
    env::split_paths(&env::var_os("PATH").unwrap_or_default())
        .map(|directory| directory.join(path))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| panic!("compiler {compiler:?} was not found in PATH"))
}
