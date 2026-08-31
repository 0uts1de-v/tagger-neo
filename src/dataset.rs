//! Dataset loading, filtering, editing, history, and persistence.
//!
//! The types in this module deliberately contain no UI state other than item
//! selection. This makes the dataset easy to use from egui (or another front
//! end) while keeping all file and edit semantics in one place.

use anyhow::{Context, Result};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

/// Image extensions understood by the dataset loader.
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

/// One image and its side-car caption file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetItem {
    /// Path to the image on disk.
    pub image_path: PathBuf,
    /// Path to the UTF-8 caption file (the image stem with a `.txt` suffix).
    pub tag_path: PathBuf,
    /// Tags in display/edit order. Tags are trimmed and never contain empty
    /// entries when loaded or changed through this module.
    pub tags: Vec<String>,
    /// Whether the item is selected in the editor.
    pub selected: bool,
    original_tags: Vec<String>,
}

impl DatasetItem {
    fn new(image_path: PathBuf, tag_path: PathBuf, tags: Vec<String>) -> Self {
        Self {
            image_path,
            tag_path,
            original_tags: tags.clone(),
            tags,
            selected: false,
        }
    }

    /// Returns true when the in-memory tags differ from the last successful
    /// load or save.
    pub fn is_modified(&self) -> bool {
        self.tags != self.original_tags
    }

    /// File name without the extension, useful for an image list row.
    pub fn stem(&self) -> String {
        self.image_path
            .file_stem()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Caption text in the format used by [`Dataset::save_all`].
    pub fn tag_text(&self) -> String {
        format_tags(&self.tags)
    }
}

/// Whether all or any include terms must be present in a caption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterMode {
    /// Every include term must match.
    And,
    /// At least one include term must match.
    Or,
}

/// Include/exclude caption filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagFilter {
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub mode: FilterMode,
}

impl Default for TagFilter {
    fn default() -> Self {
        Self {
            include: Vec::new(),
            exclude: Vec::new(),
            mode: FilterMode::And,
        }
    }
}

impl TagFilter {
    pub fn new<I, E>(include: I, exclude: E, mode: FilterMode) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
        E: IntoIterator,
        E::Item: AsRef<str>,
    {
        Self {
            include: normalize_terms(include),
            exclude: normalize_terms(exclude),
            mode,
        }
    }

    pub fn and<I, E>(include: I, exclude: E) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
        E: IntoIterator,
        E::Item: AsRef<str>,
    {
        Self::new(include, exclude, FilterMode::And)
    }

    pub fn or<I, E>(include: I, exclude: E) -> Self
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
        E: IntoIterator,
        E::Item: AsRef<str>,
    {
        Self::new(include, exclude, FilterMode::Or)
    }

    /// Tests a tag vector. Matching is case-insensitive and compares complete
    /// tags, which avoids accidentally matching `cat` in `caterpillar`.
    pub fn matches(&self, tags: &[String]) -> bool {
        let contains = |term: &str| tags.iter().any(|tag| tag.eq_ignore_ascii_case(term));

        if self.exclude.iter().any(|term| contains(term)) {
            return false;
        }
        if self.include.is_empty() {
            return true;
        }
        match self.mode {
            FilterMode::And => self.include.iter().all(|term| contains(term)),
            FilterMode::Or => self.include.iter().any(|term| contains(term)),
        }
    }
}

/// Ordering keys for [`Dataset::sort_items`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortKey {
    /// Full image path.
    Path,
    /// Image file name only.
    FileName,
    /// Number of tags in the caption.
    TagCount,
}

/// Direction for [`Dataset::sort_items`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// The key used when ordering tags inside captions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TagSortKey {
    /// Sort by the tag text.
    Alphabetical,
    /// Sort by how many times the tag occurs in the whole dataset.
    Frequency,
    /// Sort by Unicode character count.
    Length,
}

/// An edit that can be applied to multiple selected items.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchOperation {
    Append(String),
    Prepend(String),
    /// Removes complete tags equal to the supplied value (case-insensitive).
    Remove(String),
    /// Removes tags matching a regular expression.
    RemoveRegex(String),
    /// Replaces regular-expression matches inside each tag.
    Replace {
        pattern: String,
        replacement: String,
    },
    /// Replaces literal text inside each tag.
    ReplaceLiteral {
        from: String,
        to: String,
    },
    /// Removes duplicate tags while retaining their first occurrence.
    Deduplicate,
}

#[derive(Clone, Debug)]
struct TagSnapshot {
    // Key snapshots by image path rather than display index. Sorting the
    // dataset must not make undo restore one image's tags into another image.
    tags: BTreeMap<PathBuf, Vec<String>>,
}

/// Loaded image dataset and its edit history.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Directory passed to [`Dataset::load`].
    pub root: PathBuf,
    /// Items in the current display order.
    pub items: Vec<DatasetItem>,
    undo_stack: Vec<TagSnapshot>,
    redo_stack: Vec<TagSnapshot>,
}

impl Dataset {
    /// Recursively loads images below `root` in deterministic path order.
    /// Missing side-car files are treated as empty captions. Existing side-car
    /// files must be valid UTF-8.
    pub fn load<P: AsRef<Path>>(root: P) -> Result<Self> {
        Self::load_with_options(root, "txt", false)
    }

    /// Loads a dataset with a configurable caption extension and optional
    /// filename fallback for images without sidecars.
    pub fn load_with_options<P: AsRef<Path>>(
        root: P,
        caption_extension: &str,
        filename_fallback: bool,
    ) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            anyhow::bail!("dataset path is not a directory: {}", root.display());
        }

        let mut image_paths = Vec::new();
        for entry in WalkDir::new(&root).follow_links(false) {
            let entry = entry
                .with_context(|| format!("failed to walk dataset directory: {}", root.display()))?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if is_supported_image(path) {
                image_paths.push(path.to_path_buf());
            }
        }
        image_paths.sort_by(|left, right| path_ordering(left, right));

        let caption_extension = caption_extension.trim().trim_start_matches('.');
        if caption_extension.is_empty()
            || caption_extension
                .chars()
                .any(|character| matches!(character, '/' | '\\' | ':'))
        {
            anyhow::bail!("invalid caption extension: {caption_extension}");
        }

        let mut items = Vec::with_capacity(image_paths.len());
        for image_path in image_paths {
            let tag_path = image_path.with_extension(caption_extension);
            let caption_exists = tag_path.is_file();
            let tags = if caption_exists {
                let text = fs::read_to_string(&tag_path)
                    .with_context(|| format!("failed to read {} as UTF-8", tag_path.display()))?;
                parse_tags(&text)
            } else if filename_fallback {
                image_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(|stem| vec![stem.replace('_', " ")])
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let mut item = DatasetItem::new(image_path, tag_path, tags);
            if filename_fallback && !caption_exists {
                item.original_tags.clear();
            }
            items.push(item);
        }

        Ok(Self {
            root,
            items,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
        })
    }

    /// Alias for [`Dataset::load`] suitable for UI code that uses “open”.
    pub fn open<P: AsRef<Path>>(root: P) -> Result<Self> {
        Self::load(root)
    }

    pub fn open_with_options<P: AsRef<Path>>(
        root: P,
        caption_extension: &str,
        filename_fallback: bool,
    ) -> Result<Self> {
        Self::load_with_options(root, caption_extension, filename_fallback)
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn item(&self, index: usize) -> Option<&DatasetItem> {
        self.items.get(index)
    }

    pub fn item_mut(&mut self, index: usize) -> Option<&mut DatasetItem> {
        self.items.get_mut(index)
    }

    // ----- Selection -----------------------------------------------------

    pub fn set_selected(&mut self, index: usize, selected: bool) -> bool {
        if let Some(item) = self.items.get_mut(index) {
            item.selected = selected;
            true
        } else {
            false
        }
    }

    pub fn toggle_selected(&mut self, index: usize) -> bool {
        if let Some(item) = self.items.get_mut(index) {
            item.selected = !item.selected;
            item.selected
        } else {
            false
        }
    }

    pub fn select_all(&mut self) {
        for item in &mut self.items {
            item.selected = true;
        }
    }

    pub fn clear_selection(&mut self) {
        for item in &mut self.items {
            item.selected = false;
        }
    }

    pub fn selected_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| item.selected.then_some(index))
            .collect()
    }

    // ----- Filtering -----------------------------------------------------

    pub fn filtered_indices(&self, filter: &TagFilter) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| filter.matches(&item.tags).then_some(index))
            .collect()
    }

    /// Convenience form for constructing a filter without allocating a
    /// `TagFilter` at the call site.
    pub fn filter_indices(
        &self,
        include: &[String],
        exclude: &[String],
        mode: FilterMode,
    ) -> Vec<usize> {
        self.filtered_indices(&TagFilter::new(include, exclude, mode))
    }

    /// Counts tag occurrences for the supplied items.
    ///
    /// Counts are occurrence counts, rather than image counts: if a caption
    /// contains the same tag twice it contributes two to that tag's count.
    /// Invalid and duplicate indices are ignored. The returned map is sorted
    /// by tag so callers can render it deterministically.
    pub fn tag_frequencies(&self, indices: &[usize]) -> BTreeMap<String, usize> {
        let mut frequencies = BTreeMap::new();
        let mut seen = Vec::new();
        for &index in indices {
            if index >= self.items.len() || seen.contains(&index) {
                continue;
            }
            seen.push(index);
            for tag in &self.items[index].tags {
                *frequencies.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        frequencies
    }

    /// Returns tags present in every valid, unique item in `indices`.
    ///
    /// The result is alphabetical, matching the original editor's common-tag
    /// field and making the result stable across runs.
    pub fn common_tags(&self, indices: &[usize]) -> Vec<String> {
        let mut selected = Vec::new();
        for &index in indices {
            if index < self.items.len() && !selected.contains(&index) {
                selected.push(index);
            }
        }
        let Some((&first, rest)) = selected.split_first() else {
            return Vec::new();
        };

        let common = self.items[first].tags.clone();
        // Duplicate entries in one caption do not affect the common-tag
        // result, which is a set intersection in the original editor.
        let mut unique_common = Vec::with_capacity(common.len());
        for tag in common {
            if !unique_common.iter().any(|existing| existing == &tag) {
                unique_common.push(tag);
            }
        }
        let mut common = unique_common;
        common.retain(|tag| {
            rest.iter().all(|&index| {
                self.items[index]
                    .tags
                    .iter()
                    .any(|candidate| candidate == tag)
            })
        });
        common.sort();
        common
    }

    /// Replaces the common tags in each supplied caption using the same
    /// positional semantics as the original editor.
    ///
    /// Common tags are the sorted set returned by [`Dataset::common_tags`].
    /// Edited values are paired with that list: a blank value removes the
    /// corresponding common tag, omitted values remove the remaining common
    /// tags, and values beyond the common-tag list are additions. Additions
    /// are appended by default or prepended when `prepend_additions` is true.
    /// Every occurrence of a common tag in a caption is replaced, while all
    /// other tags retain their original relative order.
    pub fn replace_common_tags(
        &mut self,
        indices: &[usize],
        edited_tags: &[String],
        prepend_additions: bool,
    ) -> usize {
        let indices = self.unique_valid_indices(indices);
        if indices.is_empty() {
            return 0;
        }
        let common = self.common_tags(&indices);
        let replacements: Vec<Option<Vec<String>>> = edited_tags
            .iter()
            .map(|tag| {
                let value = tag.trim();
                if value.is_empty() {
                    None
                } else {
                    Some(parse_tags(value))
                }
            })
            .collect();
        let additions: Vec<String> = replacements
            .iter()
            .skip(common.len())
            .flat_map(|tags| tags.as_deref().unwrap_or(&[]).iter().cloned())
            .collect();

        let before = self.snapshot();
        let mut changed = 0;
        for index in indices {
            let old = self.items[index].tags.clone();
            let mut result = Vec::with_capacity(old.len() + additions.len());
            for tag in old {
                if let Some(common_index) = common.iter().position(|common| common == &tag) {
                    if let Some(Some(replacement)) = replacements.get(common_index) {
                        result.extend(replacement.iter().cloned());
                    }
                } else {
                    result.push(tag);
                }
            }
            if prepend_additions {
                let mut with_additions = additions.clone();
                with_additions.extend(result);
                result = with_additions;
            } else {
                result.extend(additions.iter().cloned());
            }
            if self.items[index].tags != result {
                self.items[index].tags = result;
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        changed
    }

    /// Sorts tags in the supplied captions. Frequency sorting uses occurrence
    /// counts from the entire dataset and breaks ties alphabetically. The
    /// operation creates one undo entry for the whole batch.
    pub fn sort_tags_in_items(
        &mut self,
        indices: &[usize],
        key: TagSortKey,
        direction: SortDirection,
    ) -> usize {
        let indices = self.unique_valid_indices(indices);
        if indices.is_empty() {
            return 0;
        }
        let frequencies = if key == TagSortKey::Frequency {
            let all_indices: Vec<usize> = (0..self.items.len()).collect();
            self.tag_frequencies(&all_indices)
        } else {
            BTreeMap::new()
        };
        let before = self.snapshot();
        let mut changed = 0;
        for index in indices {
            let old = self.items[index].tags.clone();
            let tags = &mut self.items[index].tags;
            match key {
                TagSortKey::Alphabetical => {
                    if direction == SortDirection::Ascending {
                        tags.sort();
                    } else {
                        tags.sort_by(|left, right| right.cmp(left));
                    }
                }
                TagSortKey::Frequency => {
                    tags.sort_by(|left, right| {
                        let left_count = frequencies.get(left).copied().unwrap_or(0);
                        let right_count = frequencies.get(right).copied().unwrap_or(0);
                        let count_order = left_count.cmp(&right_count);
                        if direction == SortDirection::Ascending {
                            count_order.then_with(|| left.cmp(right))
                        } else {
                            count_order.reverse().then_with(|| left.cmp(right))
                        }
                    });
                }
                TagSortKey::Length => {
                    tags.sort_by(|left, right| {
                        let order = left.chars().count().cmp(&right.chars().count());
                        if direction == SortDirection::Ascending {
                            order.then_with(|| left.cmp(right))
                        } else {
                            order.reverse().then_with(|| left.cmp(right))
                        }
                    });
                }
            }
            if *tags != old {
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        changed
    }

    /// Replaces text in the complete comma-separated caption of each supplied
    /// item. Literal replacement is used when `use_regex` is false; otherwise
    /// `search` is compiled as a regular expression. Replacement text that
    /// contains commas is split into separate tags, just like caption input.
    pub fn replace_caption(
        &mut self,
        indices: &[usize],
        search: &str,
        replacement: &str,
        use_regex: bool,
    ) -> Result<usize> {
        let indices = self.unique_valid_indices(indices);
        if indices.is_empty() || search.is_empty() {
            return Ok(0);
        }
        let regex = if use_regex {
            Some(Regex::new(search).with_context(|| {
                format!("invalid caption replacement regular expression: {search}")
            })?)
        } else {
            None
        };
        let before = self.snapshot();
        let mut changed = 0;
        for index in indices {
            let caption = self.items[index].tag_text();
            let replaced = match &regex {
                Some(regex) => regex.replace_all(&caption, replacement).into_owned(),
                None => caption.replace(search, replacement),
            };
            let tags = parse_tags(&replaced);
            if self.items[index].tags != tags {
                self.items[index].tags = tags;
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        Ok(changed)
    }

    /// Alias matching the original editor's terminology.
    pub fn search_and_replace_caption(
        &mut self,
        indices: &[usize],
        search: &str,
        replacement: &str,
        use_regex: bool,
    ) -> Result<usize> {
        self.replace_caption(indices, search, replacement, use_regex)
    }

    /// Replaces text only inside tags listed by `selected_tags`.
    ///
    /// `selected_tags` may be any iterable of string-like values, which lets
    /// callers pass either a `Vec<String>` or a `HashSet<String>` directly.
    /// Matching is exact and case-sensitive, as in the original tag editor.
    pub fn replace_selected_tags<I, S>(
        &mut self,
        indices: &[usize],
        selected_tags: I,
        search: &str,
        replacement: &str,
        use_regex: bool,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected_tags: Vec<String> = selected_tags
            .into_iter()
            .map(|tag| tag.as_ref().to_owned())
            .collect();
        let indices = self.unique_valid_indices(indices);
        if indices.is_empty() || search.is_empty() || selected_tags.is_empty() {
            return Ok(0);
        }
        let regex = if use_regex {
            Some(Regex::new(search).with_context(|| {
                format!("invalid selected-tag replacement regular expression: {search}")
            })?)
        } else {
            None
        };
        let before = self.snapshot();
        let mut changed = 0;
        for index in indices {
            let old = self.items[index].tags.clone();
            let mut tags = Vec::with_capacity(old.len());
            for tag in old {
                if selected_tags.iter().any(|selected| selected == &tag) {
                    let replaced = match &regex {
                        Some(regex) => regex.replace_all(&tag, replacement).into_owned(),
                        None => tag.replace(search, replacement),
                    };
                    tags.extend(parse_tags(&replaced));
                } else {
                    tags.push(tag);
                }
            }
            if self.items[index].tags != tags {
                self.items[index].tags = tags;
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        Ok(changed)
    }

    /// Alias matching the original editor's terminology.
    pub fn search_and_replace_selected_tags<I, S>(
        &mut self,
        indices: &[usize],
        search: &str,
        replacement: &str,
        selected_tags: I,
        use_regex: bool,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.replace_selected_tags(indices, selected_tags, search, replacement, use_regex)
    }

    /// Keeps at most `max_tags` tags in each supplied caption.
    pub fn truncate_tags(&mut self, indices: &[usize], max_tags: usize) -> usize {
        let indices = self.unique_valid_indices(indices);
        if indices.is_empty() {
            return 0;
        }
        let before = self.snapshot();
        let mut changed = 0;
        for index in indices {
            if self.items[index].tags.len() > max_tags {
                self.items[index].tags.truncate(max_tags);
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        changed
    }

    /// Alias emphasizing that the limit is a number of tags, not tokenizer
    /// tokens.
    pub fn truncate_tag_count(&mut self, indices: &[usize], max_tags: usize) -> usize {
        self.truncate_tags(indices, max_tags)
    }

    /// Filters by a comma-separated include query. A term prefixed with `-`
    /// is excluded. This is intentionally small and predictable for a search
    /// box; callers needing explicit lists can use [`Dataset::filtered_indices`].
    pub fn search(&self, query: &str, mode: FilterMode) -> Vec<usize> {
        let mut include = Vec::new();
        let mut exclude = Vec::new();
        for term in query
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
        {
            if let Some(term) = term.strip_prefix('-') {
                if !term.trim().is_empty() {
                    exclude.push(term.trim().to_owned());
                }
            } else {
                include.push(term.to_owned());
            }
        }
        self.filtered_indices(&TagFilter::new(include, exclude, mode))
    }

    // ----- Individual edits ---------------------------------------------

    /// Sets one item's tags. Empty values are discarded and surrounding
    /// whitespace is removed.
    pub fn set_tags<I, S>(&mut self, index: usize, tags: I) -> bool
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        if index >= self.items.len() {
            return false;
        }
        let tags = normalize_terms(tags);
        if self.items[index].tags == tags {
            return false;
        }
        self.record_before_change();
        self.items[index].tags = tags;
        true
    }

    /// Sets one item's tags from caption text (comma/newline separated).
    pub fn set_tag_text(&mut self, index: usize, text: &str) -> bool {
        self.set_tags(index, parse_tags(text))
    }

    /// Applies caption text updates as one undoable operation.
    pub fn set_tag_texts(&mut self, updates: &[(usize, String)]) -> usize {
        let before = self.snapshot();
        let mut changed = 0;
        for (index, text) in updates {
            let Some(item) = self.items.get_mut(*index) else {
                continue;
            };
            let tags = parse_tags(text);
            if item.tags != tags {
                item.tags = tags;
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        changed
    }

    pub fn add_tag(&mut self, index: usize, tag: &str) -> bool {
        if index >= self.items.len() {
            return false;
        }
        let additions = parse_tags(tag);
        if additions.is_empty() {
            return false;
        }
        self.record_before_change();
        self.items[index].tags.extend(additions);
        true
    }

    pub fn remove_tag(&mut self, index: usize, tag: &str) -> bool {
        if index >= self.items.len() {
            return false;
        }
        let target = tag.trim();
        if target.is_empty() {
            return false;
        }
        let before_len = self.items[index].tags.len();
        self.record_before_change();
        self.items[index]
            .tags
            .retain(|value| !value.eq_ignore_ascii_case(target));
        if self.items[index].tags.len() == before_len {
            self.discard_last_snapshot();
            false
        } else {
            true
        }
    }

    pub fn deduplicate_item(&mut self, index: usize) -> bool {
        if index >= self.items.len() {
            return false;
        }
        let before = self.items[index].tags.clone();
        let mut tags = before.clone();
        deduplicate_tags(&mut tags);
        if tags == before {
            return false;
        }
        self.record_before_change();
        self.items[index].tags = tags;
        true
    }

    // ----- Batch edits ---------------------------------------------------

    /// Applies an operation to explicit item indices. Invalid indices are
    /// ignored, so a stale selection cannot panic the UI.
    pub fn apply_operation(
        &mut self,
        indices: &[usize],
        operation: BatchOperation,
    ) -> Result<usize> {
        let prepared = PreparedOperation::new(operation)?;
        let before = self.snapshot();
        let mut changed = 0;
        let mut seen = Vec::new();
        for &index in indices {
            if index >= self.items.len() || seen.contains(&index) {
                continue;
            }
            seen.push(index);
            let tags = &mut self.items[index].tags;
            let old = tags.clone();
            prepared.apply(tags);
            if *tags != old {
                changed += 1;
            }
        }
        if changed != 0 {
            self.undo_stack.push(before);
            self.redo_stack.clear();
        }
        Ok(changed)
    }

    /// Applies an operation to the current selection. With no selected items,
    /// this is a no-op; use `select_all` when an operation should affect all.
    pub fn apply_to_selected(&mut self, operation: BatchOperation) -> Result<usize> {
        let indices = self.selected_indices();
        self.apply_operation(&indices, operation)
    }

    pub fn batch_append(&mut self, tag: &str) -> usize {
        self.apply_to_selected(BatchOperation::Append(tag.to_owned()))
            .unwrap_or(0)
    }

    pub fn batch_prepend(&mut self, tag: &str) -> usize {
        self.apply_to_selected(BatchOperation::Prepend(tag.to_owned()))
            .unwrap_or(0)
    }

    pub fn batch_remove(&mut self, tag: &str) -> usize {
        self.apply_to_selected(BatchOperation::Remove(tag.to_owned()))
            .unwrap_or(0)
    }

    pub fn batch_remove_regex(&mut self, pattern: &str) -> Result<usize> {
        self.apply_to_selected(BatchOperation::RemoveRegex(pattern.to_owned()))
    }

    /// Literal text replacement inside tags.
    pub fn batch_replace(&mut self, from: &str, to: &str) -> usize {
        self.apply_to_selected(BatchOperation::ReplaceLiteral {
            from: from.to_owned(),
            to: to.to_owned(),
        })
        .unwrap_or(0)
    }

    /// Regular-expression replacement inside tags.
    pub fn batch_replace_regex(&mut self, pattern: &str, replacement: &str) -> Result<usize> {
        self.apply_to_selected(BatchOperation::Replace {
            pattern: pattern.to_owned(),
            replacement: replacement.to_owned(),
        })
    }

    pub fn deduplicate_selected(&mut self) -> usize {
        self.apply_to_selected(BatchOperation::Deduplicate)
            .unwrap_or(0)
    }

    // ----- Sorting and dirty state --------------------------------------

    pub fn sort_items(&mut self, key: SortKey, direction: SortDirection) {
        self.items.sort_by(|left, right| {
            let ordering = match key {
                SortKey::Path => path_ordering(&left.image_path, &right.image_path),
                SortKey::FileName => left
                    .image_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .cmp(
                        &right
                            .image_path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_ascii_lowercase(),
                    ),
                SortKey::TagCount => left.tags.len().cmp(&right.tags.len()),
            };
            if direction == SortDirection::Descending {
                ordering.reverse()
            } else {
                ordering
            }
        });
    }

    pub fn sort_by_name(&mut self, ascending: bool) {
        self.sort_items(
            SortKey::FileName,
            if ascending {
                SortDirection::Ascending
            } else {
                SortDirection::Descending
            },
        );
    }

    pub fn sort_by_path(&mut self, ascending: bool) {
        self.sort_items(
            SortKey::Path,
            if ascending {
                SortDirection::Ascending
            } else {
                SortDirection::Descending
            },
        );
    }

    pub fn is_modified(&self, index: usize) -> bool {
        self.items
            .get(index)
            .map(DatasetItem::is_modified)
            .unwrap_or(false)
    }

    pub fn modified_indices(&self) -> Vec<usize> {
        self.items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| item.is_modified().then_some(index))
            .collect()
    }

    pub fn has_unsaved_changes(&self) -> bool {
        self.items.iter().any(DatasetItem::is_modified)
    }

    // ----- Persistence --------------------------------------------------

    /// Writes modified captions using a temporary file in the target directory,
    /// then renames it into place. Existing files are briefly moved aside on
    /// platforms where rename cannot replace a file (notably Windows).
    pub fn save_all(&mut self) -> Result<usize> {
        self.save_all_with_backups(false)
    }

    /// Saves modified captions and optionally keeps the previous sidecar as
    /// `<stem>.000` through `<stem>.999`, matching Dataset Tag Editor.
    pub fn save_all_with_backups(&mut self, create_backups: bool) -> Result<usize> {
        let items: Vec<(PathBuf, PathBuf, String)> = self
            .items
            .iter()
            .filter(|item| item.is_modified())
            .map(|item| {
                (
                    item.tag_path.clone(),
                    item.image_path.clone(),
                    item.tag_text(),
                )
            })
            .collect();
        for (tag_path, image_path, text) in &items {
            if create_backups && tag_path.is_file() {
                let backup = (0..1_000)
                    .map(|index| tag_path.with_extension(format!("{index:03}")))
                    .find(|path| !path.exists())
                    .with_context(|| {
                        format!("no free caption backup slot for {}", image_path.display())
                    })?;
                fs::copy(tag_path, &backup).with_context(|| {
                    format!("failed to back up caption for {}", image_path.display())
                })?;
            }
            write_atomic(tag_path, text.as_bytes())
                .with_context(|| format!("failed to save tags for {}", image_path.display()))?;
        }
        for item in self.items.iter_mut().filter(|item| item.is_modified()) {
            item.original_tags = item.tags.clone();
        }
        Ok(items.len())
    }

    /// Alias for [`Dataset::save_all`].
    pub fn save(&mut self) -> Result<usize> {
        self.save_all()
    }

    /// Marks current in-memory values as the clean baseline without writing.
    /// This is useful after an external save operation.
    pub fn mark_saved(&mut self) {
        for item in &mut self.items {
            item.original_tags = item.tags.clone();
        }
    }

    // ----- Undo / redo --------------------------------------------------

    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    pub fn undo(&mut self) -> bool {
        let previous = match self.undo_stack.pop() {
            Some(previous) => previous,
            None => return false,
        };
        let current = self.snapshot();
        self.restore(previous);
        self.redo_stack.push(current);
        true
    }

    pub fn redo(&mut self) -> bool {
        let next = match self.redo_stack.pop() {
            Some(next) => next,
            None => return false,
        };
        let current = self.snapshot();
        self.restore(next);
        self.undo_stack.push(current);
        true
    }

    pub fn clear_history(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    fn snapshot(&self) -> TagSnapshot {
        TagSnapshot {
            tags: self
                .items
                .iter()
                .map(|item| (item.image_path.clone(), item.tags.clone()))
                .collect(),
        }
    }

    fn restore(&mut self, snapshot: TagSnapshot) {
        for item in &mut self.items {
            if let Some(tags) = snapshot.tags.get(&item.image_path) {
                item.tags = tags.clone();
            }
        }
    }

    fn unique_valid_indices(&self, indices: &[usize]) -> Vec<usize> {
        let mut unique = Vec::with_capacity(indices.len());
        for &index in indices {
            if index < self.items.len() && !unique.contains(&index) {
                unique.push(index);
            }
        }
        unique
    }

    fn record_before_change(&mut self) {
        self.undo_stack.push(self.snapshot());
        self.redo_stack.clear();
    }

    fn discard_last_snapshot(&mut self) {
        self.undo_stack.pop();
    }
}

enum PreparedOperation {
    Append(Vec<String>),
    Prepend(Vec<String>),
    Remove(String),
    RemoveRegex(Regex),
    Replace(Regex, String),
    ReplaceLiteral(String, String),
    Deduplicate,
}

impl PreparedOperation {
    fn new(operation: BatchOperation) -> Result<Self> {
        Ok(match operation {
            BatchOperation::Append(value) => Self::Append(parse_tags(&value)),
            BatchOperation::Prepend(value) => Self::Prepend(parse_tags(&value)),
            BatchOperation::Remove(value) => Self::Remove(value.trim().to_owned()),
            BatchOperation::RemoveRegex(pattern) => Self::RemoveRegex(
                Regex::new(&pattern)
                    .with_context(|| format!("invalid remove regular expression: {pattern}"))?,
            ),
            BatchOperation::Replace {
                pattern,
                replacement,
            } => Self::Replace(
                Regex::new(&pattern)
                    .with_context(|| format!("invalid replace regular expression: {pattern}"))?,
                replacement,
            ),
            BatchOperation::ReplaceLiteral { from, to } => Self::ReplaceLiteral(from, to),
            BatchOperation::Deduplicate => Self::Deduplicate,
        })
    }

    fn apply(&self, tags: &mut Vec<String>) {
        match self {
            Self::Append(values) => tags.extend(values.iter().cloned()),
            Self::Prepend(values) => {
                let mut result = values.clone();
                result.extend(tags.iter().cloned());
                *tags = result;
            }
            Self::Remove(value) => {
                if value.is_empty() {
                    return;
                }
                tags.retain(|tag| !tag.eq_ignore_ascii_case(value));
            }
            Self::RemoveRegex(regex) => tags.retain(|tag| !regex.is_match(tag)),
            Self::Replace(regex, replacement) => {
                replace_each_tag(tags, |tag| regex.replace_all(tag, replacement).into_owned());
            }
            Self::ReplaceLiteral(from, to) => {
                if from.is_empty() {
                    return;
                }
                replace_each_tag(tags, |tag| tag.replace(from, to));
            }
            Self::Deduplicate => deduplicate_tags(tags),
        }
    }
}

fn replace_each_tag<F>(tags: &mut Vec<String>, mut replace: F)
where
    F: FnMut(&str) -> String,
{
    let old = std::mem::take(tags);
    let mut replaced = Vec::with_capacity(old.len());
    for tag in old {
        let value = replace(&tag);
        // A replacement may itself contain a comma, so feed it through the
        // same parser as captions rather than creating an invalid tag entry.
        replaced.extend(parse_tags(&value));
    }
    *tags = replaced;
}

fn deduplicate_tags(tags: &mut Vec<String>) {
    let mut unique = Vec::with_capacity(tags.len());
    for tag in tags.drain(..) {
        if !unique
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&tag))
        {
            unique.push(tag);
        }
    }
    *tags = unique;
}

fn parse_tags(text: &str) -> Vec<String> {
    text.split([',', '\n', '\r'])
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn format_tags(tags: &[String]) -> String {
    tags.join(", ")
}

fn normalize_terms<I, S>(terms: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    terms
        .into_iter()
        .flat_map(|term| parse_tags(term.as_ref()))
        .collect()
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|supported| extension.eq_ignore_ascii_case(supported))
        })
        .unwrap_or(false)
}

fn path_ordering(left: &Path, right: &Path) -> Ordering {
    left.to_string_lossy()
        .to_ascii_lowercase()
        .cmp(&right.to_string_lossy().to_ascii_lowercase())
}

fn write_atomic(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "tags.txt".to_owned());
    let temp = parent.join(format!(".{file_name}.tagger-neo-{nonce}.tmp"));
    let backup = parent.join(format!(".{file_name}.tagger-neo-{nonce}.bak"));

    let result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .with_context(|| format!("failed to create temporary file {}", temp.display()))?;
        file.write_all(contents)
            .with_context(|| format!("failed to write temporary file {}", temp.display()))?;
        file.flush()?;
        let _ = file.sync_all();
        drop(file);

        if path.exists() {
            fs::rename(path, &backup)
                .with_context(|| format!("failed to stage existing caption {}", path.display()))?;
        }
        match fs::rename(&temp, path) {
            Ok(()) => {
                if backup.exists() {
                    let _ = fs::remove_file(&backup);
                }
                Ok(())
            }
            Err(error) => {
                if backup.exists() {
                    if let Err(restore_error) = fs::rename(&backup, path) {
                        anyhow::bail!(
                            "failed to install {}; the original remains at {} because restoration failed: {restore_error}",
                            path.display(),
                            backup.display()
                        );
                    }
                }
                Err(error).with_context(|| format!("failed to install {}", path.display()))
            }
        }
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write_file(path: &Path, contents: &[u8]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }

    fn dataset_fixture() -> (tempfile::TempDir, Dataset) {
        let directory = tempdir().unwrap();
        write_file(&directory.path().join("z\u{00e9}.PNG"), b"not decoded here");
        write_file(
            &directory.path().join("z\u{00e9}.txt"),
            "cat, 1girl\n, cat\r\n".as_bytes(),
        );
        write_file(&directory.path().join("nested/a.jpg"), b"image");
        write_file(
            &directory.path().join("nested/a.txt"),
            "blue hair, smile".as_bytes(),
        );
        write_file(&directory.path().join("ignored.gif"), b"gif");
        let dataset = Dataset::load(directory.path()).unwrap();
        (directory, dataset)
    }

    #[test]
    fn recursively_loads_supported_images_and_captions() {
        let (_directory, dataset) = dataset_fixture();
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.items[0].stem(), "a");
        assert_eq!(dataset.items[0].tags, vec!["blue hair", "smile"]);
        assert_eq!(dataset.items[1].tags, vec!["cat", "1girl", "cat"]);
        assert!(dataset.items.iter().all(|item| !item.is_modified()));
    }

    #[test]
    fn missing_caption_is_empty_and_invalid_utf8_is_error() {
        let directory = tempdir().unwrap();
        write_file(&directory.path().join("empty.webp"), b"image");
        assert!(Dataset::load(directory.path()).unwrap().items[0]
            .tags
            .is_empty());
        write_file(&directory.path().join("bad.txt"), &[0xff, 0xfe]);
        write_file(&directory.path().join("bad.png"), b"image");
        assert!(Dataset::load(directory.path()).is_err());
    }

    #[test]
    fn custom_caption_extension_and_filename_fallback_are_supported() {
        let directory = tempdir().unwrap();
        write_file(&directory.path().join("blue_hair.png"), b"image");
        let fallback = Dataset::load_with_options(directory.path(), "caption", true).unwrap();
        assert_eq!(fallback.items[0].tags, vec!["blue hair"]);
        assert!(fallback.items[0].is_modified());
        write_file(
            &directory.path().join("blue_hair.caption"),
            b"1girl\nsolo, smile",
        );
        let loaded = Dataset::load_with_options(directory.path(), ".caption", false).unwrap();
        assert_eq!(loaded.items[0].tags, vec!["1girl", "solo", "smile"]);
    }

    #[test]
    fn selection_and_filters_support_and_or_and_exclusion() {
        let (_directory, mut dataset) = dataset_fixture();
        assert!(dataset.set_selected(0, true));
        assert_eq!(dataset.selected_indices(), vec![0]);
        assert_eq!(
            dataset.filtered_indices(&TagFilter::and(["blue hair"], std::iter::empty::<&str>(),)),
            vec![0]
        );
        assert_eq!(
            dataset.filtered_indices(&TagFilter::and(
                ["cat", "1girl"],
                std::iter::empty::<&str>(),
            )),
            vec![1]
        );
        assert!(dataset.filtered_indices(&TagFilter::or(["cat", "smile"], ["cat"])) == vec![0]);
        assert!(!dataset.toggle_selected(0));
    }

    #[test]
    fn individual_edit_and_dirty_tracking() {
        let (_directory, mut dataset) = dataset_fixture();
        assert!(dataset.set_tag_text(0, " new, tag "));
        assert_eq!(dataset.items[0].tags, vec!["new", "tag"]);
        assert!(dataset.is_modified(0));
        assert!(!dataset.set_tag_text(0, "new, tag"));
        assert!(dataset.add_tag(0, "third"));
        assert!(dataset.remove_tag(0, "THIRD"));
        assert!(dataset.deduplicate_item(1));
        assert_eq!(dataset.items[1].tags, vec!["cat", "1girl"]);
        dataset.mark_saved();
        assert!(!dataset.has_unsaved_changes());
    }

    #[test]
    fn batch_operations_and_regex_are_applied_to_selection() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.select_all();
        assert_eq!(dataset.batch_prepend("quality"), 2);
        assert_eq!(dataset.batch_append("score_9, "), 2);
        assert_eq!(dataset.batch_remove("smile"), 1);
        assert_eq!(
            dataset
                .batch_replace_regex(r"score_(\d)", "score_${1}_high")
                .unwrap(),
            2
        );
        assert_eq!(dataset.batch_remove_regex(r"^1girl$").unwrap(), 1);
        assert!(dataset.items[0].tags.contains(&"score_9_high".to_owned()));
        assert!(!dataset.items[0].tags.contains(&"smile".to_owned()));
        assert!(!dataset.items[1].tags.contains(&"1girl".to_owned()));
    }

    #[test]
    fn undo_redo_tracks_only_real_changes_and_new_edits_clear_redo() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.select_all();
        assert_eq!(dataset.batch_append("x"), 2);
        assert!(dataset.can_undo());
        assert!(dataset.undo());
        assert!(!dataset.items[0].tags.contains(&"x".to_owned()));
        assert!(dataset.redo());
        assert!(dataset.items[0].tags.contains(&"x".to_owned()));
        assert!(dataset.set_tag_text(0, "different"));
        assert!(!dataset.can_redo());
        assert_eq!(
            dataset
                .apply_to_selected(BatchOperation::Remove("absent".to_owned()))
                .unwrap(),
            0
        );
    }

    #[test]
    fn sorting_and_search_are_deterministic() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.sort_by_name(false);
        assert_eq!(dataset.items[0].stem(), "z\u{00e9}");
        assert_eq!(
            dataset.search("cat, -1girl", FilterMode::And),
            Vec::<usize>::new()
        );
        assert_eq!(dataset.search("cat, smile", FilterMode::Or), vec![0, 1]);
    }

    #[test]
    fn save_all_writes_captions_and_clears_dirty_state() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.set_tag_text(0, "one, two");
        let count = dataset.save_all().unwrap();
        assert_eq!(count, 1);
        assert_eq!(
            fs::read_to_string(&dataset.items[0].tag_path).unwrap(),
            "one, two"
        );
        assert!(!dataset.has_unsaved_changes());
        assert!(fs::read_dir(dataset.items[0].tag_path.parent().unwrap())
            .unwrap()
            .all(|entry| {
                let name = entry.unwrap().file_name().to_string_lossy().into_owned();
                !name.contains("tagger-neo-")
            }));
    }

    #[test]
    fn optional_save_backup_uses_numbered_sidecar() {
        let (_directory, mut dataset) = dataset_fixture();
        let old = fs::read_to_string(&dataset.items[0].tag_path).unwrap();
        dataset.set_tag_text(0, "changed");
        assert_eq!(dataset.save_all_with_backups(true).unwrap(), 1);
        let backup = dataset.items[0].tag_path.with_extension("000");
        assert_eq!(fs::read_to_string(backup).unwrap(), old);
        assert_eq!(
            fs::read_to_string(&dataset.items[0].tag_path).unwrap(),
            "changed"
        );
    }

    #[test]
    fn invalid_regex_does_not_change_history_or_tags() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.select_all();
        let before = dataset.items[0].tags.clone();
        assert!(dataset.batch_replace_regex("[", "x").is_err());
        assert_eq!(dataset.items[0].tags, before);
        assert!(!dataset.can_undo());
    }

    #[test]
    fn frequencies_and_common_tags_are_deterministic() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.items[0].tags = vec!["z".into(), "a".into(), "z".into(), "b".into()];
        dataset.items[1].tags = vec!["b".into(), "a".into(), "d".into()];
        let frequencies = dataset.tag_frequencies(&[0, 1, 1, 99]);
        assert_eq!(frequencies.get("a"), Some(&2));
        assert_eq!(frequencies.get("b"), Some(&2));
        assert_eq!(frequencies.get("z"), Some(&2));
        assert_eq!(frequencies.get("d"), Some(&1));
        assert_eq!(dataset.common_tags(&[0, 1]), vec!["a", "b"]);
        assert!(dataset.common_tags(&[]).is_empty());
    }

    #[test]
    fn common_tag_replacement_preserves_positions_and_has_one_undo_step() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.items[0].tags = vec!["A".into(), "A".into(), "B".into(), "C".into()];
        dataset.items[1].tags = vec!["A".into(), "B".into(), "D".into()];
        dataset.mark_saved();

        assert_eq!(
            dataset.replace_common_tags(&[0, 1], &["X".into(), "Y".into(), "Z".into()], false,),
            2
        );
        assert_eq!(dataset.items[0].tags, vec!["X", "X", "Y", "C", "Z"]);
        assert_eq!(dataset.items[1].tags, vec!["X", "Y", "D", "Z"]);
        assert!(dataset.undo());
        assert_eq!(dataset.items[0].tags, vec!["A", "A", "B", "C"]);
        assert_eq!(dataset.items[1].tags, vec!["A", "B", "D"]);
        assert!(!dataset.can_undo());

        assert_eq!(
            dataset.replace_common_tags(&[0, 1], &["".into(), "Q".into()], true,),
            2
        );
        assert_eq!(dataset.items[0].tags, vec!["Q", "C"]);
        assert_eq!(dataset.items[1].tags, vec!["Q", "D"]);
    }

    #[test]
    fn tag_sorting_uses_dataset_frequency_and_one_undo_step() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.items[0].tags = vec!["z".into(), "a".into(), "z".into()];
        dataset.items[1].tags = vec!["a".into(), "b".into()];
        dataset.mark_saved();

        assert_eq!(
            dataset.sort_tags_in_items(&[0, 1], TagSortKey::Frequency, SortDirection::Ascending,),
            2
        );
        assert_eq!(dataset.items[0].tags, vec!["a", "z", "z"]);
        assert_eq!(dataset.items[1].tags, vec!["b", "a"]);
        assert!(dataset.undo());
        assert_eq!(dataset.items[0].tags, vec!["z", "a", "z"]);
        assert_eq!(dataset.items[1].tags, vec!["a", "b"]);
    }

    #[test]
    fn caption_and_selected_tag_replacement_support_literal_and_regex() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.items[0].tags = vec!["1boy".into(), "blue hair".into(), "1boy, extra".into()];
        dataset.items[1].tags = vec!["1boy".into(), "red hair".into()];
        dataset.mark_saved();

        assert_eq!(
            dataset
                .replace_caption(&[0, 1], "boy", "girl", false)
                .unwrap(),
            2
        );
        assert_eq!(
            dataset.items[0].tags,
            vec!["1girl", "blue hair", "1girl", "extra"]
        );
        assert!(dataset.undo());

        assert_eq!(
            dataset
                .replace_selected_tags(
                    &[0, 1],
                    [&"1boy".to_owned(), &"blue hair".to_owned()],
                    r"^(1)boy$|^blue",
                    "${1}girl",
                    true,
                )
                .unwrap(),
            2
        );
        assert_eq!(
            dataset.items[0].tags,
            vec!["1girl", "girl hair", "1boy, extra"]
        );
        assert_eq!(dataset.items[1].tags, vec!["1girl", "red hair"]);
        assert!(dataset.undo());
        assert_eq!(dataset.items[0].tags[0], "1boy");
    }

    #[test]
    fn truncate_tag_count_limits_tags_and_undoes_as_one_batch() {
        let (_directory, mut dataset) = dataset_fixture();
        dataset.items[0].tags = vec!["a".into(), "b".into(), "c".into()];
        dataset.items[1].tags = vec!["d".into()];
        dataset.mark_saved();
        assert_eq!(dataset.truncate_tags(&[0, 1], 2), 1);
        assert_eq!(dataset.items[0].tags, vec!["a", "b"]);
        assert!(!dataset.items[1].is_modified());
        assert!(dataset.undo());
        assert_eq!(dataset.items[0].tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn undo_after_sort_restores_tags_by_image_path() {
        let (_directory, mut dataset) = dataset_fixture();
        let first_path = dataset.items[0].image_path.clone();
        let second_path = dataset.items[1].image_path.clone();
        dataset.set_tag_text(0, "edited");
        dataset.sort_by_name(false);
        assert!(dataset.undo());
        let first = dataset
            .items
            .iter()
            .find(|item| item.image_path == first_path)
            .unwrap();
        let second = dataset
            .items
            .iter()
            .find(|item| item.image_path == second_path)
            .unwrap();
        assert_eq!(first.tags, vec!["blue hair", "smile"]);
        assert_eq!(second.tags, vec!["cat", "1girl", "cat"]);
    }
}
