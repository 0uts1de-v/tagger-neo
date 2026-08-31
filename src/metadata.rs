//! Kohya_ss-compatible dataset metadata import and export.
//!
//! Kohya metadata files are JSON objects whose image records are usually
//! keyed by a stem (for example, `image_001`) or by an absolute image path.
//! This module deliberately keeps the JSON values opaque so that fields added
//! by kohya or another tool (hashes, resolutions, dataset statistics, and so
//! on) survive a merge export unchanged.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Which field should be written to each metadata record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetadataExportMode {
    /// Write the editor's tag string to the `tags` field.
    #[default]
    Tags,
    /// Write the editor's tag string to the `caption` field.
    Caption,
}

/// How an export interacts with an existing JSON file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetadataWriteMode {
    /// Preserve existing records and fields, updating only the selected field
    /// for records supplied to the export.
    #[default]
    Merge,
    /// Replace the complete JSON object with the supplied image records.
    Overwrite,
}

/// How image paths become keys during export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetadataKeyMode {
    /// Use the image filename without its extension (kohya's common format).
    #[default]
    Stem,
    /// Use the normalized absolute image path.
    AbsolutePath,
}

/// Options for [`write_metadata`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetadataExportOptions {
    pub mode: MetadataExportMode,
    pub write_mode: MetadataWriteMode,
    pub key_mode: MetadataKeyMode,
}

impl MetadataExportOptions {
    /// Export `tags` records while preserving existing metadata.
    pub const fn tags() -> Self {
        Self {
            mode: MetadataExportMode::Tags,
            write_mode: MetadataWriteMode::Merge,
            key_mode: MetadataKeyMode::Stem,
        }
    }

    /// Export `caption` records while preserving existing metadata.
    pub const fn captions() -> Self {
        Self {
            mode: MetadataExportMode::Caption,
            write_mode: MetadataWriteMode::Merge,
            key_mode: MetadataKeyMode::Stem,
        }
    }
}

/// The JSON object stored by a kohya metadata file.
///
/// Values are retained as [`serde_json::Value`] instead of being deserialized
/// into a narrow record type. This keeps unrelated kohya fields intact when a
/// file is merged and also accepts both image records and root-level metadata
/// such as `ss_tag_frequency`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct KohyaMetadata {
    #[serde(flatten)]
    pub records: BTreeMap<String, Value>,
}

/// Alias with a more generic name for callers that do not need kohya-specific
/// terminology.
pub type MetadataDocument = KohyaMetadata;

impl KohyaMetadata {
    /// Returns a record by its exact JSON key.
    pub fn record(&self, key: &str) -> Option<&Value> {
        self.records.get(key)
    }

    /// Returns the number of root entries, including kohya bookkeeping fields.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns whether the JSON object has no entries.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Finds the best matching metadata key for an image path.
    ///
    /// Absolute-path keys are preferred over filename keys, which are in turn
    /// preferred over stems. This means datasets containing duplicate stems
    /// still resolve correctly when their metadata uses absolute paths.
    pub fn key_for_image<P: AsRef<Path>>(&self, image_path: P) -> Option<&str> {
        let image_path = image_path.as_ref();
        self.records
            .keys()
            .filter_map(|key| path_match_score(key, image_path).map(|score| (score, key)))
            .max_by(|(left_score, left_key), (right_score, right_key)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| right_key.cmp(left_key))
            })
            .map(|(_, key)| key.as_str())
    }

    /// Returns the raw `tags` value for an image, if it is a string, array, or
    /// frequency-map object.
    pub fn raw_tags_for_image<P: AsRef<Path>>(&self, image_path: P) -> Option<String> {
        let key = self.key_for_image(image_path)?;
        let record = self.records.get(key)?;
        record_text_field(record, "tags")
    }

    /// Returns the raw `caption` value for an image.
    pub fn caption_for_image<P: AsRef<Path>>(&self, image_path: P) -> Option<String> {
        let key = self.key_for_image(image_path)?;
        let record = self.records.get(key)?;
        record_text_field(record, "caption")
    }

    /// Returns the editor-ready tag string for an image.
    ///
    /// When both fields exist, kohya's caption is placed first and its tags
    /// follow it, separated by `, `. Empty or non-string fields are ignored.
    pub fn tags_for_image<P: AsRef<Path>>(&self, image_path: P) -> Option<String> {
        let key = self.key_for_image(image_path)?;
        let record = self.records.get(key)?;
        record_tag_string(record)
    }
}

/// Reads one kohya metadata JSON file.
pub fn read_metadata<P: AsRef<Path>>(path: P) -> Result<KohyaMetadata> {
    let path = path.as_ref();
    let file = fs::File::open(path)
        .with_context(|| format!("failed to open metadata file {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse metadata JSON {}", path.display()))
}

/// Returns an empty document when `path` does not exist, otherwise reads it.
/// This is convenient for a merge export whose destination may be new.
pub fn read_metadata_or_empty<P: AsRef<Path>>(path: P) -> Result<KohyaMetadata> {
    let path = path.as_ref();
    if path.exists() {
        read_metadata(path)
    } else {
        Ok(KohyaMetadata::default())
    }
}

/// Writes image tag strings to a kohya metadata JSON file.
///
/// Each item is `(image_path, tag_string)`. Tags are written as a JSON array;
/// captions are written as a string. In merge mode, existing fields
/// and records are retained; in overwrite mode, only the supplied records are
/// emitted. The write uses a temporary file and a rollback backup on Windows.
pub fn write_metadata<P, I, Q, T>(
    metadata_path: P,
    entries: I,
    options: MetadataExportOptions,
) -> Result<usize>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = (Q, T)>,
    Q: AsRef<Path>,
    T: AsRef<str>,
{
    let metadata_path = metadata_path.as_ref();
    let mut document = if options.write_mode == MetadataWriteMode::Merge {
        read_metadata_or_empty(metadata_path)?
    } else {
        KohyaMetadata::default()
    };

    let mut count = 0;
    let mut exported_keys = HashSet::new();
    for (image_path, tag_string) in entries {
        let key = metadata_key(image_path.as_ref(), options.key_mode)?;
        if !exported_keys.insert(key.clone()) {
            bail!(
                "duplicate metadata key `{key}`; use absolute-path keys for datasets with duplicate stems"
            );
        }
        let record = document
            .records
            .entry(key)
            .or_insert_with(|| Value::Object(Map::new()));
        if !record.is_object() {
            *record = Value::Object(Map::new());
        }
        let object = record
            .as_object_mut()
            .expect("metadata record was normalized to an object");
        let (field, value) = match options.mode {
            MetadataExportMode::Tags => (
                "tags",
                Value::Array(
                    tag_string
                        .as_ref()
                        .split([',', '\n', '\r'])
                        .map(str::trim)
                        .filter(|tag| !tag.is_empty())
                        .map(|tag| Value::String(tag.to_owned()))
                        .collect(),
                ),
            ),
            MetadataExportMode::Caption => (
                "caption",
                Value::String(tag_string.as_ref().trim().to_owned()),
            ),
        };
        object.insert(field.to_owned(), value);
        count += 1;
    }

    let mut json =
        serde_json::to_vec_pretty(&document).context("failed to serialize kohya metadata JSON")?;
    json.push(b'\n');
    write_json_atomic(metadata_path, &json)?;
    Ok(count)
}

/// Compatibility alias for callers that prefer an export-oriented name.
pub fn export_metadata<P, I, Q, T>(
    metadata_path: P,
    entries: I,
    options: MetadataExportOptions,
) -> Result<usize>
where
    P: AsRef<Path>,
    I: IntoIterator<Item = (Q, T)>,
    Q: AsRef<Path>,
    T: AsRef<str>,
{
    write_metadata(metadata_path, entries, options)
}

/// Combines optional caption and tags strings in the same order used by
/// [`KohyaMetadata::tags_for_image`].
pub fn combine_caption_and_tags(caption: Option<&str>, tags: Option<&str>) -> Option<String> {
    let caption = caption.map(str::trim).filter(|text| !text.is_empty());
    let tags = tags.map(str::trim).filter(|text| !text.is_empty());
    match (caption, tags) {
        (Some(caption), Some(tags)) => Some(format!("{caption}, {tags}")),
        (Some(caption), None) => Some(caption.to_owned()),
        (None, Some(tags)) => Some(tags.to_owned()),
        (None, None) => None,
    }
}

/// Extracts the editor-ready tag string from one JSON record.
pub fn record_tag_string(record: &Value) -> Option<String> {
    let caption = record_text_field(record, "caption");
    let tags = record_text_field(record, "tags");
    combine_caption_and_tags(caption.as_deref(), tags.as_deref())
}

fn record_text_field(record: &Value, field: &str) -> Option<String> {
    let value = record.as_object()?.get(field)?;
    match value {
        Value::String(text) => nonempty_text(text),
        Value::Array(values) => {
            let parts = values
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty());
            let text = parts.collect::<Vec<_>>().join(", ");
            nonempty_text(&text)
        }
        Value::Object(values) if field == "tags" => {
            let text = values
                .keys()
                .map(String::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            nonempty_text(&text)
        }
        _ => None,
    }
}

fn nonempty_text(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn metadata_key(path: &Path, key_mode: MetadataKeyMode) -> Result<String> {
    match key_mode {
        MetadataKeyMode::Stem => path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .filter(|stem| !stem.is_empty())
            .context("cannot use an image path without a filename stem as metadata key"),
        MetadataKeyMode::AbsolutePath => Ok(normalized_absolute_path(path)),
    }
}

fn path_match_score(key: &str, image_path: &Path) -> Option<u8> {
    let key = key.trim();
    if key.is_empty() {
        return None;
    }

    let absolute_image = normalized_absolute_path(image_path);
    if is_absolute_path(key)
        && paths_equal(&normalized_absolute_path(Path::new(key)), &absolute_image)
    {
        return Some(100);
    }

    let image_file_name = image_path.file_name()?.to_string_lossy();
    let image_stem = image_path.file_stem()?.to_string_lossy();
    if text_equal(key, &image_stem) {
        return Some(90);
    }
    if text_equal(key, &image_file_name) {
        return Some(80);
    }

    let key_path = Path::new(key);
    if key_path
        .file_stem()
        .map(|stem| text_equal(&stem.to_string_lossy(), &image_stem))
        == Some(true)
        && key_path.extension().is_some()
    {
        return Some(70);
    }
    if key_path
        .file_name()
        .map(|name| text_equal(&name.to_string_lossy(), &image_file_name))
        == Some(true)
    {
        return Some(60);
    }
    None
}

fn normalized_absolute_path(path: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    let mut text = absolute.to_string_lossy().replace('\\', "/");
    while text.len() > 1 && text.ends_with('/') {
        text.pop();
    }
    #[cfg(windows)]
    {
        text.make_ascii_lowercase();
    }
    text
}

fn is_absolute_path(path: &str) -> bool {
    Path::new(path).is_absolute()
        || (path.as_bytes().get(1) == Some(&b':')
            && path.as_bytes().first().is_some_and(u8::is_ascii_alphabetic))
}

fn paths_equal(left: &str, right: &str) -> bool {
    #[cfg(windows)]
    {
        left.eq_ignore_ascii_case(right)
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

fn text_equal(left: &str, right: &str) -> bool {
    #[cfg(windows)]
    {
        left.eq_ignore_ascii_case(right)
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

fn write_json_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.is_dir() {
        anyhow::bail!(
            "metadata destination directory does not exist: {}",
            parent.display()
        );
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| std::borrow::Cow::Borrowed("metadata.json"));
    let temp_path = parent.join(format!(".{file_name}.{nonce}.tmp"));
    let backup_path = parent.join(format!(".{file_name}.{nonce}.bak"));

    let result = (|| -> Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
            .with_context(|| {
                format!(
                    "failed to create temporary metadata file {}",
                    temp_path.display()
                )
            })?;
        let mut writer = BufWriter::new(file);
        writer
            .write_all(bytes)
            .context("failed to write metadata JSON")?;
        writer.flush().context("failed to flush metadata JSON")?;
        let file = writer
            .into_inner()
            .map_err(|error| anyhow::anyhow!("failed to finalize metadata JSON: {error}"))?;
        file.sync_all().context("failed to sync metadata JSON")?;

        let had_original = path.exists();
        if had_original {
            fs::rename(path, &backup_path).with_context(|| {
                format!("failed to stage existing metadata file {}", path.display())
            })?;
        }
        match fs::rename(&temp_path, path) {
            Ok(()) => {
                if had_original {
                    let _ = fs::remove_file(&backup_path);
                }
                Ok(())
            }
            Err(error) => {
                if had_original {
                    if let Err(restore_error) = fs::rename(&backup_path, path) {
                        anyhow::bail!(
                            "failed to replace {}; the original remains at {} because restoration failed: {restore_error}",
                            path.display(),
                            backup_path.display()
                        );
                    }
                }
                Err(error)
                    .with_context(|| format!("failed to replace metadata file {}", path.display()))
            }
        }
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn reads_stem_records_and_combines_caption_before_tags() {
        let directory = tempdir().unwrap();
        let metadata_path = directory.path().join("meta.json");
        fs::write(
            &metadata_path,
            r#"{
                "image_001": {"caption":"a caption", "tags":"one, two", "hash":"keep"},
                "image_002": {"tags":{"three":1,"four":1}},
                "image_003": {"caption":"caption only"}
            }"#,
        )
        .unwrap();

        let metadata = read_metadata(&metadata_path).unwrap();
        assert_eq!(
            metadata.tags_for_image(directory.path().join("image_001.png")),
            Some("a caption, one, two".to_owned())
        );
        assert_eq!(
            metadata.tags_for_image(directory.path().join("image_002.jpg")),
            Some("four, three".to_owned())
        );
        assert_eq!(
            metadata.tags_for_image(directory.path().join("image_003.webp")),
            Some("caption only".to_owned())
        );
        assert_eq!(
            metadata.tags_for_image(directory.path().join("missing.png")),
            None
        );
    }

    #[test]
    fn absolute_path_keys_take_precedence_over_stems() {
        let directory = tempdir().unwrap();
        let image_path = directory.path().join("image.png");
        let absolute_key = normalized_absolute_path(&image_path);
        let metadata = KohyaMetadata {
            records: BTreeMap::from([
                ("image".to_owned(), serde_json::json!({"tags":"stem"})),
                (absolute_key.clone(), serde_json::json!({"tags":"absolute"})),
            ]),
        };
        assert_eq!(
            metadata.key_for_image(&image_path),
            Some(absolute_key.as_str())
        );
        assert_eq!(
            metadata.tags_for_image(&image_path),
            Some("absolute".to_owned())
        );
    }

    #[test]
    fn overwrite_and_merge_preserve_expected_fields() {
        let directory = tempdir().unwrap();
        let metadata_path = directory.path().join("meta.json");
        fs::write(
            &metadata_path,
            r#"{
                "old": {"tags":"untouched", "custom":true},
                "keep": {"caption":"keep caption", "custom":7}
            }"#,
        )
        .unwrap();

        let image = directory.path().join("new.png");
        let count = write_metadata(
            &metadata_path,
            [(&image, "caption value")],
            MetadataExportOptions {
                mode: MetadataExportMode::Caption,
                write_mode: MetadataWriteMode::Merge,
                key_mode: MetadataKeyMode::Stem,
            },
        )
        .unwrap();
        assert_eq!(count, 1);
        let merged = read_metadata(&metadata_path).unwrap();
        assert_eq!(merged.record("old").unwrap()["custom"], true);
        assert_eq!(merged.record("new").unwrap()["caption"], "caption value");

        write_metadata(
            &metadata_path,
            [(&image, "new tags")],
            MetadataExportOptions {
                mode: MetadataExportMode::Tags,
                write_mode: MetadataWriteMode::Overwrite,
                key_mode: MetadataKeyMode::Stem,
            },
        )
        .unwrap();
        let overwritten = read_metadata(&metadata_path).unwrap();
        assert_eq!(overwritten.len(), 1);
        assert_eq!(
            overwritten.record("new").unwrap()["tags"],
            serde_json::json!(["new tags"])
        );
        assert!(overwritten.record("old").is_none());
    }

    #[test]
    fn atomic_write_leaves_valid_json_and_supports_empty_merge() {
        let directory = tempdir().unwrap();
        let metadata_path = directory.path().join("nested").join("meta.json");
        assert!(write_metadata(
            &metadata_path,
            std::iter::empty::<(PathBuf, String)>(),
            MetadataExportOptions::default(),
        )
        .is_err());

        let metadata_path = directory.path().join("meta.json");
        write_metadata(
            &metadata_path,
            [(directory.path().join("one.png"), "one".to_owned())],
            MetadataExportOptions::default(),
        )
        .unwrap();
        let text = fs::read_to_string(&metadata_path).unwrap();
        assert!(text.ends_with('\n'));
        assert_eq!(
            read_metadata(&metadata_path)
                .unwrap()
                .tags_for_image(directory.path().join("one.png")),
            Some("one".to_owned())
        );
    }

    #[test]
    fn duplicate_stem_keys_are_rejected_without_writing() {
        let directory = tempdir().unwrap();
        let metadata_path = directory.path().join("meta.json");
        let result = write_metadata(
            &metadata_path,
            [
                (directory.path().join("a/same.png"), "one"),
                (directory.path().join("b/same.jpg"), "two"),
            ],
            MetadataExportOptions::default(),
        );
        assert!(result.is_err());
        assert!(!metadata_path.exists());
    }
}
