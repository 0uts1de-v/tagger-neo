use std::{env, fs, io, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_CFG_WINDOWS").is_none() {
        return;
    }

    if let Err(error) = materialize_directml() {
        panic!("failed to package DirectML.dll: {error}");
    }
}

/// ort-sys uses a symlink when Windows Developer Mode permits it. Such a link
/// can point into a build-user cache and break as soon as the executable is
/// copied or launched outside that environment, so turn it into a real DLL.
fn materialize_directml() -> io::Result<()> {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    let artifact_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("Cargo OUT_DIR has the expected layout");

    for directory in [artifact_dir.to_path_buf(), artifact_dir.join("deps")] {
        let dll = directory.join("DirectML.dll");
        let metadata = match fs::symlink_metadata(&dll) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error),
        };
        if !metadata.file_type().is_symlink() {
            continue;
        }

        let source = fs::canonicalize(&dll)?;
        let staged = directory.join("DirectML.tagger-neo.tmp");
        fs::copy(source, &staged)?;
        fs::remove_file(&dll)?;
        fs::rename(staged, dll)?;
    }
    Ok(())
}
