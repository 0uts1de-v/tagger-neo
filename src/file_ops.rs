//! Safe file-group operations used by the move/delete dataset controls.
//!
//! A dataset item is made up of an image, its caption sidecar, and optional
//! caption backups. Backups use the convention from the original Dataset Tag
//! Editor: the caption stem followed by a three digit extension from `.000`
//! through `.999` (for example, `image.007`).

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};

static COPY_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Number of backup suffixes supported by the original editor.
pub const CAPTION_BACKUP_COUNT: usize = 1_000;

/// Return the existing caption backup files for `tag_path` in numeric order.
///
/// The returned paths use the caption path's stem and replace its extension,
/// so `foo.bar.txt` is associated with `foo.bar.000` through `foo.bar.999`.
/// Missing backups are omitted. Directories with a matching name are not
/// treated as backups and are never touched by the file-operation helpers.
pub fn existing_caption_backups(tag_path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let tag_path = tag_path.as_ref();
    if tag_path.file_name().is_none() {
        bail!("caption path has no file name: {}", tag_path.display());
    }

    let mut backups = Vec::new();
    for index in 0..CAPTION_BACKUP_COUNT {
        let backup = tag_path.with_extension(format!("{index:03}"));
        if is_regular_file(&backup)? {
            backups.push(backup);
        }
    }
    Ok(backups)
}

/// Move the requested members of one image/caption file group.
///
/// `image_path` and `tag_path` are explicit paths because callers may use a
/// caption extension different from `.txt`. Existing destination files are
/// always an error; no files are changed until every requested source and
/// destination has passed the preflight checks. The return value is the
/// number of files moved.
pub fn move_file_group(
    image_path: impl AsRef<Path>,
    tag_path: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    move_image: bool,
    move_caption: bool,
    move_backups: bool,
) -> Result<usize> {
    move_file_groups(
        &[(
            image_path.as_ref().to_path_buf(),
            tag_path.as_ref().to_path_buf(),
        )],
        destination,
        move_image,
        move_caption,
        move_backups,
    )
}

/// Move several image/caption groups as one preflighted operation.
pub fn move_file_groups(
    groups: &[(PathBuf, PathBuf)],
    destination: impl AsRef<Path>,
    move_image: bool,
    move_caption: bool,
    move_backups: bool,
) -> Result<usize> {
    let destination = destination.as_ref();

    let mut sources = Vec::new();
    let mut seen_sources = HashSet::new();
    for (image_path, tag_path) in groups {
        for source in
            selected_existing_files(image_path, tag_path, move_image, move_caption, move_backups)?
        {
            if seen_sources.insert(source.clone()) {
                sources.push(source);
            }
        }
    }
    if sources.is_empty() {
        return Ok(0);
    }

    if destination.exists() {
        if !destination.is_dir() {
            bail!("destination is not a directory: {}", destination.display());
        }
    } else {
        fs::create_dir_all(destination).with_context(|| {
            format!(
                "failed to create destination directory {}",
                destination.display()
            )
        })?;
    }

    let mut destinations = HashSet::new();
    let mut plan = Vec::with_capacity(sources.len());
    for source in sources {
        let name = source
            .file_name()
            .ok_or_else(|| anyhow!("source path has no file name: {}", source.display()))?;
        let target = destination.join(name);
        if !destinations.insert(target.clone()) {
            bail!(
                "multiple selected files have the same destination: {}",
                target.display()
            );
        }
        if path_exists(&target) {
            bail!("destination file already exists: {}", target.display());
        }
        plan.push((source, target));
    }

    let mut completed: Vec<(PathBuf, PathBuf)> = Vec::new();
    for (source, target) in &plan {
        if let Err(error) = move_one_file(source, target) {
            let mut rollback_errors = Vec::new();
            for (old_source, old_target) in completed.iter().rev() {
                if let Err(rollback_error) = move_one_file(old_target, old_source) {
                    rollback_errors.push(format!("{rollback_error:#}"));
                }
            }
            if rollback_errors.is_empty() {
                return Err(error).context("move failed; completed files were restored");
            }
            bail!(
                "move failed: {error:#}; rollback also failed: {}",
                rollback_errors.join(" | ")
            );
        }
        completed.push((source.clone(), target.clone()));
    }
    Ok(plan.len())
}

/// Delete the requested members of one image/caption file group.
///
/// Only individual regular files are removed. Missing files and matching
/// directories are ignored, while filesystem errors are returned to the
/// caller. The return value is the number of files deleted.
pub fn delete_file_group(
    image_path: impl AsRef<Path>,
    tag_path: impl AsRef<Path>,
    delete_image: bool,
    delete_caption: bool,
    delete_backups: bool,
) -> Result<usize> {
    delete_file_groups(
        &[(
            image_path.as_ref().to_path_buf(),
            tag_path.as_ref().to_path_buf(),
        )],
        delete_image,
        delete_caption,
        delete_backups,
    )
}

/// Delete several image/caption groups after validating every target.
pub fn delete_file_groups(
    groups: &[(PathBuf, PathBuf)],
    delete_image: bool,
    delete_caption: bool,
    delete_backups: bool,
) -> Result<usize> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    for (image_path, tag_path) in groups {
        for path in selected_existing_files(
            image_path,
            tag_path,
            delete_image,
            delete_caption,
            delete_backups,
        )? {
            if seen.insert(path.clone()) {
                files.push(path);
            }
        }
    }

    let mut deleted = 0;
    for path in files {
        fs::remove_file(&path)
            .with_context(|| format!("failed to delete file {}", path.display()))?;
        deleted += 1;
    }
    Ok(deleted)
}

fn selected_existing_files(
    image_path: &Path,
    tag_path: &Path,
    include_image: bool,
    include_caption: bool,
    include_backups: bool,
) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();
    let mut push_file = |path: &Path| -> Result<()> {
        // is_file deliberately excludes directories. It still permits a
        // symlink whose target is a regular file; remove_file/rename act on
        // the symlink itself, which is the least surprising safe behavior.
        if is_regular_file(path)? && seen.insert(path.to_path_buf()) {
            files.push(path.to_path_buf());
        }
        Ok(())
    };

    if include_image {
        push_file(image_path)?;
    }
    if include_caption {
        push_file(tag_path)?;
    }
    if include_backups {
        for backup in existing_caption_backups(tag_path)? {
            push_file(&backup)?;
        }
    }
    Ok(files)
}

fn move_one_file(source: &Path, target: &Path) -> Result<()> {
    match fs::rename(source, target) {
        Ok(()) => return Ok(()),
        Err(error) if is_cross_device(&error) => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to move {} to {}",
                    source.display(),
                    target.display()
                )
            })
        }
    }

    copy_then_remove(source, target)
}

/// Copy a file without replacing a destination, verify the complete byte
/// count, then remove the source. This is the safe fallback for a move across
/// volumes (where rename cannot work).
fn copy_then_remove(source: &Path, target: &Path) -> Result<()> {
    let metadata = fs::metadata(source)
        .with_context(|| format!("failed to inspect source {}", source.display()))?;
    if !metadata.is_file() {
        bail!("source is not a regular file: {}", source.display());
    }

    let mut input = File::open(source)
        .with_context(|| format!("failed to open source {}", source.display()))?;
    let temp_target = temporary_copy_path(target)?;
    let mut output = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_target)
    {
        Ok(file) => file,
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to create temporary destination {}",
                    temp_target.display()
                )
            })
        }
    };

    let copy_result = (|| -> io::Result<u64> {
        let copied = io::copy(&mut input, &mut output)?;
        output.flush()?;
        output.sync_all()?;
        Ok(copied)
    })();
    drop(output);

    let copied = match copy_result {
        Ok(copied) if copied == metadata.len() => copied,
        Ok(copied) => {
            let _ = fs::remove_file(&temp_target);
            bail!(
                "copy verification failed for {} (expected {} bytes, copied {})",
                source.display(),
                metadata.len(),
                copied
            );
        }
        Err(error) => {
            let _ = fs::remove_file(&temp_target);
            return Err(error).with_context(|| {
                format!(
                    "failed while copying {} to {}",
                    source.display(),
                    target.display()
                )
            });
        }
    };

    if let Err(error) = fs::set_permissions(&temp_target, metadata.permissions()) {
        let _ = fs::remove_file(&temp_target);
        return Err(error)
            .with_context(|| format!("failed to preserve permissions on {}", target.display()));
    }

    debug_assert_eq!(copied, metadata.len());
    if let Err(error) = fs::rename(&temp_target, target) {
        let _ = fs::remove_file(&temp_target);
        return Err(error)
            .with_context(|| format!("failed to install copied file {}", target.display()));
    }
    if let Err(error) = fs::remove_file(source) {
        let _ = fs::remove_file(target);
        return Err(error).with_context(|| {
            format!(
                "copied {} to {}, but failed to remove the source",
                source.display(),
                target.display()
            )
        });
    }
    Ok(())
}

fn temporary_copy_path(target: &Path) -> Result<PathBuf> {
    let file_name = target
        .file_name()
        .ok_or_else(|| anyhow!("destination path has no file name: {}", target.display()))?
        .to_string_lossy();
    let sequence = COPY_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(target.with_file_name(format!(
        ".{file_name}.tagger-neo-copy-{}-{sequence}.tmp",
        process::id()
    )))
}

fn is_cross_device(error: &io::Error) -> bool {
    // EXDEV on Unix and ERROR_NOT_SAME_DEVICE on Windows. ErrorKind was
    // intentionally avoided here so this remains compatible with the crate's
    // Rust 1.88 MSRV on every supported target.
    matches!(error.raw_os_error(), Some(17 | 18))
}

fn path_exists(path: &Path) -> bool {
    // `Path::exists` follows links and reports false for a dangling link.
    // symlink_metadata treats that link as an occupied destination, avoiding
    // an accidental replacement or an opaque platform-specific error.
    fs::symlink_metadata(path).is_ok()
}

fn is_regular_file(path: &Path) -> Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) => Ok(metadata.is_file()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    #[test]
    fn enumerates_existing_backups_in_numeric_order() {
        let dir = tempdir().unwrap();
        let tag = dir.path().join("foo.bar.txt");
        write(&tag, b"caption");
        write(&dir.path().join("foo.bar.007"), b"7");
        write(&dir.path().join("foo.bar.000"), b"0");
        write(&dir.path().join("foo.bar.999"), b"999");
        write(&dir.path().join("foo.bar.1000"), b"ignored");
        fs::create_dir(dir.path().join("foo.bar.008")).unwrap();

        let backups = existing_caption_backups(&tag).unwrap();
        let names: Vec<_> = backups
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["foo.bar.000", "foo.bar.007", "foo.bar.999"]);
    }

    #[test]
    fn move_group_preserves_names_and_selected_members() {
        let root = tempdir().unwrap();
        let source = root.path().join("source");
        let destination = root.path().join("destination");
        fs::create_dir(&source).unwrap();
        let image = source.join("sample.png");
        let tag = source.join("sample.txt");
        let backup = source.join("sample.001");
        write(&image, b"image");
        write(&tag, b"tag");
        write(&backup, b"backup");

        let moved = move_file_group(&image, &tag, &destination, true, true, true).unwrap();
        assert_eq!(moved, 3);
        assert!(!image.exists());
        assert!(!tag.exists());
        assert!(!backup.exists());
        assert_eq!(fs::read(destination.join("sample.png")).unwrap(), b"image");
        assert_eq!(fs::read(destination.join("sample.txt")).unwrap(), b"tag");
        assert_eq!(fs::read(destination.join("sample.001")).unwrap(), b"backup");
    }

    #[test]
    fn move_preflights_collisions_without_partial_changes() {
        let root = tempdir().unwrap();
        let source = root.path().join("source");
        let destination = root.path().join("destination");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&destination).unwrap();
        let image = source.join("sample.png");
        let tag = source.join("sample.txt");
        write(&image, b"image");
        write(&tag, b"tag");
        write(&destination.join("sample.txt"), b"existing");

        let error = move_file_group(&image, &tag, &destination, true, true, false).unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert!(image.exists());
        assert!(tag.exists());
        assert_eq!(
            fs::read(destination.join("sample.txt")).unwrap(),
            b"existing"
        );
        assert!(!destination.join("sample.png").exists());
    }

    #[test]
    fn multi_group_move_preflights_duplicate_names_before_moving() {
        let root = tempdir().unwrap();
        let first = root.path().join("first");
        let second = root.path().join("second");
        let destination = root.path().join("destination");
        let first_image = first.join("same.png");
        let second_image = second.join("same.png");
        write(&first_image, b"first");
        write(&second_image, b"second");

        let groups = vec![
            (first_image.clone(), first.join("same.txt")),
            (second_image.clone(), second.join("same.txt")),
        ];
        let error = move_file_groups(&groups, &destination, true, false, false).unwrap_err();

        assert!(error.to_string().contains("same destination"));
        assert_eq!(fs::read(&first_image).unwrap(), b"first");
        assert_eq!(fs::read(&second_image).unwrap(), b"second");
        assert!(!destination.join("same.png").exists());
    }

    #[test]
    fn delete_group_removes_only_requested_members() {
        let dir = tempdir().unwrap();
        let image = dir.path().join("sample.png");
        let tag = dir.path().join("sample.txt");
        let backup = dir.path().join("sample.004");
        write(&image, b"image");
        write(&tag, b"tag");
        write(&backup, b"backup");

        let deleted = delete_file_group(&image, &tag, false, true, true).unwrap();
        assert_eq!(deleted, 2);
        assert!(image.exists());
        assert!(!tag.exists());
        assert!(!backup.exists());
    }

    #[test]
    fn directories_are_never_deleted_or_moved() {
        let root = tempdir().unwrap();
        let image = root.path().join("sample.png");
        let tag = root.path().join("sample.txt");
        fs::create_dir(&image).unwrap();
        fs::create_dir(&tag).unwrap();

        assert_eq!(
            delete_file_group(&image, &tag, true, true, true).unwrap(),
            0
        );
        assert_eq!(
            move_file_group(&image, &tag, root.path().join("dest"), true, true, true).unwrap(),
            0
        );
        assert!(image.is_dir());
        assert!(tag.is_dir());
    }
}
