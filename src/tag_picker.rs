//! A small, reusable tag chooser for egui panels.
//!
//! The picker deliberately keeps the data model separate from the dataset.  A
//! caller supplies `(tag, frequency)` pairs for the current image selection,
//! and can then use [`TagPicker::selected_tags`] as input to a batch operation
//! or an image filter.

use eframe::egui::{self, Color32, RichText, Ui};
use regex::Regex;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Ordering used by [`TagPicker`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TagSort {
    /// Sort by the tag text, case-insensitively.
    #[default]
    Alpha,
    /// Sort by the number of captions containing the tag.
    Frequency,
    /// Sort by the number of Unicode scalar values in the tag.
    Length,
}

/// A compatibility name that reads naturally at call sites that use a
/// generic "sort mode" setting.
pub type SortMode = TagSort;

/// Compatibility name used by the reference editor's Python implementation.
pub type SortBy = TagSort;

/// Information returned by one call to [`TagPicker::show`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TagPickerResponse {
    /// True when the selected set changed during this frame.
    pub changed: bool,
    /// The tag whose chip was clicked during this frame, if any.
    pub clicked: Option<String>,
    /// Number of tags currently matching the controls.
    pub visible_count: usize,
}

/// Search, sort, and select a list of tags with compact clickable chips.
///
/// `prefix` and `suffix` are intentionally booleans, matching the original
/// Dataset Tag Editor controls: when either is enabled the query is matched at
/// that edge.  If both are enabled they are combined with OR semantics.
#[derive(Clone, Debug)]
pub struct TagPicker {
    query: String,
    prefix: bool,
    suffix: bool,
    regex: bool,
    sort: TagSort,
    descending: bool,
    selected: HashSet<String>,
}

impl Default for TagPicker {
    fn default() -> Self {
        Self {
            query: String::new(),
            prefix: false,
            suffix: false,
            regex: false,
            sort: TagSort::Alpha,
            descending: false,
            selected: HashSet::new(),
        }
    }
}

impl TagPicker {
    /// Construct a picker with the default alpha/ascending ordering.
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the picker and return changes made by this frame.
    pub fn show(&mut self, ui: &mut Ui, tags: &[(String, usize)]) -> TagPickerResponse {
        let mut response = TagPickerResponse::default();

        ui.horizontal(|ui| {
            let search = ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .desired_width(180.0)
                    .hint_text("⌕"),
            );
            search.on_hover_text("Search tags");

            let prefix = ui
                .toggle_value(&mut self.prefix, "↤")
                .on_hover_text("Prefix match");
            let suffix = ui
                .toggle_value(&mut self.suffix, "↦")
                .on_hover_text("Suffix match");
            let regex = ui
                .toggle_value(&mut self.regex, ".*")
                .on_hover_text("Regular expression");
            // Keep the three controls' responses live so an egui caller can
            // inspect the row with a debugger without relying on labels.
            let _ = (prefix, suffix, regex);

            ui.separator();
            ui.selectable_value(&mut self.sort, TagSort::Alpha, "A")
                .on_hover_text("Alphabetical");
            ui.selectable_value(&mut self.sort, TagSort::Frequency, "#")
                .on_hover_text("Frequency");
            ui.selectable_value(&mut self.sort, TagSort::Length, "↕")
                .on_hover_text("Length");
            ui.selectable_value(&mut self.descending, true, "↓")
                .on_hover_text("Descending")
                .on_hover_ui(|ui| {
                    if self.descending {
                        ui.label("↓");
                    }
                });
            ui.selectable_value(&mut self.descending, false, "↑")
                .on_hover_text("Ascending");
        });

        let visible = self.visible_entries(tags);
        response.visible_count = visible.len();

        ui.horizontal(|ui| {
            let selected_label = format!("☑ {}", self.selected.len());
            if ui
                .button(selected_label)
                .on_hover_text("Select visible tags")
                .clicked()
            {
                response.changed = self.select_visible_entries(&visible);
            }
            if ui.button("☐").on_hover_text("Clear visible tags").clicked() {
                response.changed = self.clear_visible_entries(&visible) || response.changed;
            }
            if self.regex && self.regex_error().is_some() {
                ui.colored_label(Color32::from_rgb(235, 110, 110), "⚠")
                    .on_hover_text(self.regex_error().unwrap_or_default());
            }
            ui.label(RichText::new(format!("{}/{}", self.selected.len(), tags.len())).weak());
        });

        egui::ScrollArea::vertical()
            .id_source(("tag-picker-tags", self as *const Self as usize))
            .max_height(220.0)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (tag, count) in &visible {
                        let selected = self.selected.contains(tag);
                        let mut label = RichText::new(format!("{tag}  {count}")).size(12.0);
                        if selected {
                            label = label.strong();
                        }
                        if ui.selectable_label(selected, label).clicked() {
                            if selected {
                                self.selected.remove(tag);
                            } else {
                                self.selected.insert(tag.clone());
                            }
                            response.changed = true;
                            response.clicked = Some(tag.clone());
                        }
                    }
                });
            });

        response
    }

    /// Alias for [`TagPicker::show`] for callers that use egui's `ui` naming.
    pub fn ui(&mut self, ui: &mut Ui, tags: &[(String, usize)]) -> TagPickerResponse {
        self.show(ui, tags)
    }

    /// Current search expression.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Replace the search expression.
    pub fn set_query(&mut self, query: impl Into<String>) {
        self.query = query.into();
    }

    pub fn prefix(&self) -> bool {
        self.prefix
    }

    pub fn set_prefix(&mut self, enabled: bool) {
        self.prefix = enabled;
    }

    pub fn suffix(&self) -> bool {
        self.suffix
    }

    pub fn set_suffix(&mut self, enabled: bool) {
        self.suffix = enabled;
    }

    pub fn regex(&self) -> bool {
        self.regex
    }

    pub fn set_regex(&mut self, enabled: bool) {
        self.regex = enabled;
    }

    /// Whether the current query is a valid regular expression.
    pub fn regex_valid(&self) -> bool {
        self.regex_error().is_none()
    }

    /// Return the parser error for the current query, if it is invalid.
    pub fn regex_error_message(&self) -> Option<String> {
        self.regex_error()
    }

    pub fn sort(&self) -> TagSort {
        self.sort
    }

    pub fn set_sort(&mut self, sort: TagSort) {
        self.sort = sort;
    }

    pub fn descending(&self) -> bool {
        self.descending
    }

    pub fn set_descending(&mut self, descending: bool) {
        self.descending = descending;
    }

    /// Return a copy so callers cannot mutate picker state without going
    /// through the selection methods.
    pub fn selected_tags(&self) -> HashSet<String> {
        self.selected.clone()
    }

    /// Borrow the current selection without allocating.
    pub fn selected_tags_ref(&self) -> &HashSet<String> {
        &self.selected
    }

    /// Replace the current selection.
    pub fn set_selected_tags<I, S>(&mut self, tags: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.selected = tags.into_iter().map(Into::into).collect();
    }

    /// Remove all selected tags.
    pub fn clear_selected_tags(&mut self) {
        self.selected.clear();
    }

    /// Return sorted, filtered `(tag, frequency)` pairs for the supplied data.
    pub fn visible_entries(&self, tags: &[(String, usize)]) -> Vec<(String, usize)> {
        let mut visible: Vec<_> = tags
            .iter()
            .filter(|(tag, _)| self.matches(tag))
            .cloned()
            .collect();
        visible.sort_by(|a, b| self.compare_entries(a, b));
        visible
    }

    /// Return only the tag text for the current filtered/sorted view.
    pub fn filtered_visible_tags(&self, tags: &[(String, usize)]) -> Vec<String> {
        self.visible_entries(tags)
            .into_iter()
            .map(|(tag, _)| tag)
            .collect()
    }

    /// Short alias for [`TagPicker::filtered_visible_tags`].
    pub fn visible_tags(&self, tags: &[(String, usize)]) -> Vec<String> {
        self.filtered_visible_tags(tags)
    }

    /// Select every tag in the current filtered view.
    pub fn select_visible(&mut self, tags: &[(String, usize)]) -> bool {
        let visible = self.visible_entries(tags);
        self.select_visible_entries(&visible)
    }

    /// Clear every tag in the current filtered view.
    pub fn clear_visible(&mut self, tags: &[(String, usize)]) -> bool {
        let visible = self.visible_entries(tags);
        self.clear_visible_entries(&visible)
    }

    fn select_visible_entries(&mut self, visible: &[(String, usize)]) -> bool {
        let before = self.selected.len();
        self.selected
            .extend(visible.iter().map(|(tag, _)| tag.clone()));
        self.selected.len() != before
    }

    fn clear_visible_entries(&mut self, visible: &[(String, usize)]) -> bool {
        let before = self.selected.len();
        for (tag, _) in visible {
            self.selected.remove(tag);
        }
        self.selected.len() != before
    }

    fn regex_error(&self) -> Option<String> {
        if !self.regex || self.query.is_empty() {
            return None;
        }
        Regex::new(&self.query).err().map(|error| error.to_string())
    }

    fn matches(&self, tag: &str) -> bool {
        if self.query.is_empty() {
            return true;
        }

        if self.regex {
            let Ok(regex) = Regex::new(&self.query) else {
                // This mirrors the reference editor: a malformed expression
                // must not hide every tag while the user is fixing it.
                return true;
            };
            let prefix = self.prefix
                && Regex::new(&format!(r"^(?:{})", self.query))
                    .map(|pattern| pattern.is_match(tag))
                    .unwrap_or(true);
            let suffix = self.suffix
                && Regex::new(&format!(r"(?:{})$", self.query))
                    .map(|pattern| pattern.is_match(tag))
                    .unwrap_or(true);
            return prefix || suffix || (!self.prefix && !self.suffix && regex.is_match(tag));
        }

        (self.prefix && tag.starts_with(&self.query))
            || (self.suffix && tag.ends_with(&self.query))
            || (!self.prefix && !self.suffix && tag.contains(&self.query))
    }

    fn compare_entries(&self, a: &(String, usize), b: &(String, usize)) -> Ordering {
        let primary = match self.sort {
            TagSort::Alpha => a.0.to_lowercase().cmp(&b.0.to_lowercase()),
            TagSort::Frequency => a.1.cmp(&b.1),
            TagSort::Length => a.0.chars().count().cmp(&b.0.chars().count()),
        };
        let tie =
            a.0.to_lowercase()
                .cmp(&b.0.to_lowercase())
                .then(a.0.cmp(&b.0));
        let ordering = primary.then(tie);
        if self.descending {
            ordering.reverse()
        } else {
            ordering
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TagPicker, TagSort};

    fn tags() -> Vec<(String, usize)> {
        [
            ("blue eyes", 4),
            ("red eyes", 2),
            ("blue hair", 9),
            ("1girl", 7),
        ]
        .into_iter()
        .map(|(tag, count)| (tag.to_owned(), count))
        .collect()
    }

    #[test]
    fn substring_prefix_and_suffix_matching() {
        let data = tags();
        let mut picker = TagPicker::new();
        picker.set_query("blue");
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["blue eyes", "blue hair"]
        );

        picker.set_prefix(true);
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["blue eyes", "blue hair"]
        );
        picker.set_query("eyes");
        picker.set_prefix(false);
        picker.set_suffix(true);
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["blue eyes", "red eyes"]
        );
    }

    #[test]
    fn regex_and_invalid_regex_are_safe() {
        let data = tags();
        let mut picker = TagPicker::new();
        picker.set_regex(true);
        picker.set_query(r"^blue .+");
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["blue eyes", "blue hair"]
        );
        picker.set_query("[");
        assert_eq!(picker.filtered_visible_tags(&data).len(), data.len());
    }

    #[test]
    fn sorting_is_deterministic_and_supports_descending() {
        let data = tags();
        let mut picker = TagPicker::new();
        picker.set_sort(TagSort::Frequency);
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["red eyes", "blue eyes", "1girl", "blue hair"]
        );
        picker.set_descending(true);
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["blue hair", "1girl", "blue eyes", "red eyes"]
        );

        picker.set_sort(TagSort::Length);
        picker.set_descending(false);
        assert_eq!(
            picker.filtered_visible_tags(&data),
            vec!["1girl", "red eyes", "blue eyes", "blue hair"]
        );
    }

    #[test]
    fn visible_selection_preserves_hidden_tags() {
        let data = tags();
        let mut picker = TagPicker::new();
        picker.set_selected_tags(["1girl", "not-visible"]);
        picker.set_query("blue");
        assert!(picker.select_visible(&data));
        assert!(picker.selected_tags().contains("1girl"));
        assert!(picker.selected_tags().contains("not-visible"));
        assert!(picker.selected_tags().contains("blue eyes"));
        assert!(picker.clear_visible(&data));
        assert_eq!(picker.selected_tags().len(), 2);
    }
}
