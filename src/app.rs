//! Native egui application.

use crate::dataset::{BatchOperation, Dataset, FilterMode, SortDirection, TagSortKey};
use crate::file_ops::{delete_file_groups, move_file_groups};
use crate::metadata::{
    read_metadata, write_metadata, MetadataExportMode, MetadataExportOptions, MetadataKeyMode,
    MetadataWriteMode,
};
use crate::tag_picker::TagPicker;
use crate::tagger::{
    ensure_model_files_for, model_is_available, model_is_available_for, TagCategory, TagPrediction,
    TaggerOptions, Wd14Model, Wd14Tagger,
};
use eframe::egui::{self, Color32, ColorImage, RichText, TextureHandle, TextureOptions};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const THUMB: f32 = 150.0;
const THUMB_CARD_WIDTH: f32 = THUMB + 10.0;
const THUMB_IMAGE_HEIGHT: f32 = THUMB - 8.0;
const THUMB_TAGS_HEIGHT: f32 = 42.0;
const THUMB_CARD_MARGIN: f32 = 10.0;
const COMPACT_LAYOUT_WIDTH: f32 = 720.0;
const PANE_GUTTER: f32 = 8.0;
const MAX_CACHED_THUMBNAILS: usize = 192;
const PREFS_KEY: &str = "tagger-neo.preferences.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
struct Preferences {
    dark_mode: bool,
    wd14_model: Wd14Model,
    general: f32,
    character: f32,
    rating: bool,
    append: bool,
    spaces: bool,
    character_first: bool,
    undesired: String,
    backup_on_save: bool,
    caption_extension: String,
    filename_fallback: bool,
    truncate_count: usize,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            dark_mode: true,
            wd14_model: Wd14Model::default(),
            general: 0.35,
            character: 0.35,
            rating: false,
            append: false,
            spaces: true,
            character_first: false,
            undesired: String::new(),
            backup_on_save: true,
            caption_extension: "txt".to_owned(),
            filename_fallback: false,
            truncate_count: 75,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Panel {
    Edit,
    Filter,
    Batch,
    Wd14,
    Files,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Batch {
    Common,
    Append,
    Prepend,
    Remove,
    Replace,
    Dedupe,
    Sort,
    Truncate,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum BatchTarget {
    Visible,
    Checked,
    Current,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReplaceTarget {
    Selected,
    Each,
    Caption,
}
#[derive(Clone)]
struct DeletePlan {
    files: Vec<(PathBuf, PathBuf)>,
    image: bool,
    caption: bool,
    backups: bool,
}
enum Event {
    Progress(String, f32),
    Predictions(PathBuf, Vec<TagPrediction>),
    Done,
    Cancelled,
    Error(String),
}

pub struct TaggerNeoApp {
    dark_mode: bool,
    wd14_model: Wd14Model,
    data: Option<Dataset>,
    current: Option<usize>,
    include: String,
    exclude: String,
    mode: FilterMode,
    negative_mode: FilterMode,
    positive_filter_enabled: bool,
    negative_filter_enabled: bool,
    selection_filter: bool,
    filter_positive: bool,
    panel: Panel,
    caption: String,
    caption_for: Option<usize>,
    batch: Batch,
    batch_target: BatchTarget,
    replace_target: ReplaceTarget,
    batch_a: String,
    batch_b: String,
    regex: bool,
    common_source: String,
    common_edit: String,
    common_prepend: bool,
    tag_sort_key: TagSortKey,
    tag_sort_direction: SortDirection,
    truncate_count: usize,
    positive_picker: TagPicker,
    negative_picker: TagPicker,
    edit_picker: TagPicker,
    batch_picker: TagPicker,
    textures: HashMap<PathBuf, TextureHandle>,
    texture_order: VecDeque<PathBuf>,
    broken: HashSet<PathBuf>,
    status: String,
    general: f32,
    character: f32,
    rating: bool,
    append: bool,
    spaces: bool,
    character_first: bool,
    undesired: String,
    worker: Option<Receiver<Event>>,
    pending_predictions: Vec<(PathBuf, Vec<TagPrediction>)>,
    cancel: Option<Arc<AtomicBool>>,
    progress: f32,
    progress_text: String,
    confirm_open: bool,
    confirm_exit: bool,
    pending_delete: Option<DeletePlan>,
    allow_exit: bool,
    files_image: bool,
    files_caption: bool,
    files_backups: bool,
    move_destination: String,
    backup_on_save: bool,
    metadata_mode: MetadataExportMode,
    metadata_write: MetadataWriteMode,
    metadata_key: MetadataKeyMode,
    caption_extension: String,
    loaded_caption_extension: String,
    filename_fallback: bool,
}

impl TaggerNeoApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_fonts(&cc.egui_ctx);
        apply_theme(&cc.egui_ctx, true);
        let mut style = (*cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(8.0, 7.0);
        style.visuals.window_rounding = egui::Rounding::same(8.0);
        cc.egui_ctx.set_style(style);
        let mut app = Self {
            dark_mode: true,
            wd14_model: Wd14Model::default(),
            data: None,
            current: None,
            include: String::new(),
            exclude: String::new(),
            mode: FilterMode::And,
            negative_mode: FilterMode::Or,
            positive_filter_enabled: true,
            negative_filter_enabled: true,
            selection_filter: false,
            filter_positive: true,
            panel: Panel::Edit,
            caption: String::new(),
            caption_for: None,
            batch: Batch::Common,
            batch_target: BatchTarget::Visible,
            replace_target: ReplaceTarget::Selected,
            batch_a: String::new(),
            batch_b: String::new(),
            regex: false,
            common_source: String::new(),
            common_edit: String::new(),
            common_prepend: false,
            tag_sort_key: TagSortKey::Alphabetical,
            tag_sort_direction: SortDirection::Ascending,
            truncate_count: 75,
            positive_picker: TagPicker::new(),
            negative_picker: TagPicker::new(),
            edit_picker: TagPicker::new(),
            batch_picker: TagPicker::new(),
            textures: HashMap::new(),
            texture_order: VecDeque::new(),
            broken: HashSet::new(),
            status: String::new(),
            general: 0.35,
            character: 0.35,
            rating: false,
            append: false,
            spaces: true,
            character_first: false,
            undesired: String::new(),
            worker: None,
            pending_predictions: Vec::new(),
            cancel: None,
            progress: 0.0,
            progress_text: String::new(),
            confirm_open: false,
            confirm_exit: false,
            pending_delete: None,
            allow_exit: false,
            files_image: true,
            files_caption: true,
            files_backups: true,
            move_destination: String::new(),
            backup_on_save: true,
            metadata_mode: MetadataExportMode::Tags,
            metadata_write: MetadataWriteMode::Merge,
            metadata_key: MetadataKeyMode::Stem,
            caption_extension: "txt".to_owned(),
            loaded_caption_extension: "txt".to_owned(),
            filename_fallback: false,
        };
        if let Some(preferences) = cc
            .storage
            .and_then(|storage| eframe::get_value::<Preferences>(storage, PREFS_KEY))
        {
            app.apply_preferences(preferences);
        }
        apply_theme(&cc.egui_ctx, app.dark_mode);
        if let Some(path) = std::env::args_os().nth(1).map(PathBuf::from) {
            match Dataset::open_with_options(&path, &app.caption_extension, app.filename_fallback) {
                Ok(data) => {
                    let count = data.len();
                    app.data = Some(data);
                    app.current = (count > 0).then_some(0);
                    app.status = format!("{} · {count}", path.display());
                    app.loaded_caption_extension = app.caption_extension.clone();
                }
                Err(error) => app.status = format!("⚠ {error:#}"),
            }
        }
        app
    }

    fn preferences(&self) -> Preferences {
        Preferences {
            dark_mode: self.dark_mode,
            wd14_model: self.wd14_model,
            general: self.general,
            character: self.character,
            rating: self.rating,
            append: self.append,
            spaces: self.spaces,
            character_first: self.character_first,
            undesired: self.undesired.clone(),
            backup_on_save: self.backup_on_save,
            caption_extension: self.caption_extension.clone(),
            filename_fallback: self.filename_fallback,
            truncate_count: self.truncate_count,
        }
    }

    fn apply_preferences(&mut self, preferences: Preferences) {
        self.dark_mode = preferences.dark_mode;
        self.wd14_model = preferences.wd14_model;
        self.general = preferences.general.clamp(0.0, 1.0);
        self.character = preferences.character.clamp(0.0, 1.0);
        self.rating = preferences.rating;
        self.append = preferences.append;
        self.spaces = preferences.spaces;
        self.character_first = preferences.character_first;
        self.undesired = preferences.undesired;
        self.backup_on_save = preferences.backup_on_save;
        self.caption_extension = preferences.caption_extension;
        self.filename_fallback = preferences.filename_fallback;
        self.truncate_count = preferences.truncate_count;
    }

    fn cache() -> PathBuf {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("tagger-neo")
            .join("models")
            .join("wd14")
    }

    fn open(&mut self) {
        if self.worker.is_some() {
            self.status = "⚠ WD14 …".to_owned();
            return;
        }
        let Some(path) = rfd::FileDialog::new().pick_folder() else {
            return;
        };
        match Dataset::open_with_options(&path, &self.caption_extension, self.filename_fallback) {
            Ok(data) => {
                let count = data.len();
                self.data = Some(data);
                self.current = (count > 0).then_some(0);
                self.caption_for = None;
                self.textures.clear();
                self.texture_order.clear();
                self.broken.clear();
                self.status = format!("{} · {count}", path.display());
                self.loaded_caption_extension = self.caption_extension.clone();
            }
            Err(e) => self.status = format!("⚠ {e:#}"),
        }
    }

    fn request_open(&mut self) {
        if self.worker.is_some() {
            self.status = "⚠ WD14 …".to_owned();
            return;
        }
        self.commit_caption();
        if self
            .data
            .as_ref()
            .map(Dataset::has_unsaved_changes)
            .unwrap_or(false)
        {
            self.confirm_open = true;
        } else {
            self.open();
        }
    }

    fn save(&mut self) {
        self.commit_caption();
        if self.data.is_some()
            && self
                .loaded_caption_extension
                .trim_start_matches('.')
                .ne(self.caption_extension.trim().trim_start_matches('.'))
        {
            self.status = "⚠ .ext ↻".to_owned();
            return;
        }
        if let Some(data) = &mut self.data {
            match data.save_all_with_backups(self.backup_on_save) {
                Ok(n) => self.status = format!("✓ {n}"),
                Err(e) => self.status = format!("⚠ {e:#}"),
            }
        }
    }

    fn terms(text: &str, picker: &TagPicker) -> Vec<String> {
        let mut terms: Vec<String> = text
            .split(',')
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(str::to_owned)
            .collect();
        for tag in picker.selected_tags_ref() {
            if !terms.iter().any(|term| term.eq_ignore_ascii_case(tag)) {
                terms.push(tag.clone());
            }
        }
        terms
    }

    fn visible(&self) -> Vec<usize> {
        let Some(data) = &self.data else {
            return Vec::new();
        };
        let include = if self.positive_filter_enabled {
            Self::terms(&self.include, &self.positive_picker)
        } else {
            Vec::new()
        };
        let exclude = if self.negative_filter_enabled {
            Self::terms(&self.exclude, &self.negative_picker)
        } else {
            Vec::new()
        };
        data.items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                if self.selection_filter && !item.selected {
                    return None;
                }
                let has = |term: &str| item.tags.iter().any(|tag| tag.eq_ignore_ascii_case(term));
                let included = include.is_empty()
                    || match self.mode {
                        FilterMode::And => include.iter().all(|term| has(term)),
                        FilterMode::Or => include.iter().any(|term| has(term)),
                    };
                let excluded = !exclude.is_empty()
                    && match self.negative_mode {
                        FilterMode::And => exclude.iter().all(|term| has(term)),
                        FilterMode::Or => exclude.iter().any(|term| has(term)),
                    };
                (included && !excluded).then_some(index)
            })
            .collect()
    }

    fn tag_stats(&self, indices: &[usize]) -> Vec<(String, usize)> {
        self.data
            .as_ref()
            .map(|data| data.tag_frequencies(indices).into_iter().collect())
            .unwrap_or_default()
    }

    fn all_indices(&self) -> Vec<usize> {
        self.data
            .as_ref()
            .map(|data| (0..data.len()).collect())
            .unwrap_or_default()
    }

    fn batch_indices(&self) -> Vec<usize> {
        match self.batch_target {
            BatchTarget::Visible => self.visible(),
            BatchTarget::Checked => self
                .data
                .as_ref()
                .map(Dataset::selected_indices)
                .unwrap_or_default(),
            BatchTarget::Current => self.current.into_iter().collect(),
        }
    }

    fn sync_caption(&mut self) {
        if self.caption_for == self.current {
            return;
        }
        self.caption_for = self.current;
        self.caption = self
            .current
            .and_then(|i| self.data.as_ref()?.item(i))
            .map(|x| x.tag_text())
            .unwrap_or_default();
    }

    fn commit_caption(&mut self) {
        if let Some(index) = self.caption_for {
            if let Some(data) = &mut self.data {
                data.set_tag_text(index, &self.caption);
            }
        }
    }

    fn texture(&mut self, ctx: &egui::Context, path: &Path) -> Option<TextureHandle> {
        if let Some(t) = self.textures.get(path) {
            return Some(t.clone());
        }
        if self.broken.contains(path) {
            return None;
        }
        match image::open(path) {
            Ok(image) => {
                let image = image.thumbnail(384, 384).to_rgba8();
                let size = [image.width() as usize, image.height() as usize];
                let color = ColorImage::from_rgba_unmultiplied(size, image.as_raw());
                let t = ctx.load_texture(path.to_string_lossy(), color, TextureOptions::LINEAR);
                self.textures.insert(path.to_owned(), t.clone());
                self.texture_order.push_back(path.to_owned());
                while self.texture_order.len() > MAX_CACHED_THUMBNAILS {
                    if let Some(expired) = self.texture_order.pop_front() {
                        self.textures.remove(&expired);
                    }
                }
                Some(t)
            }
            Err(_) => {
                self.broken.insert(path.to_owned());
                None
            }
        }
    }

    fn download(&mut self, ctx: egui::Context) {
        if self.worker.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.worker = Some(rx);
        let model = self.wd14_model;
        std::thread::spawn(move || {
            let result = ensure_model_files_for(Self::cache(), model, &mut |p| {
                let v = p
                    .total
                    .map(|t| p.downloaded as f32 / t as f32)
                    .unwrap_or(0.0);
                let _ = tx.send(Event::Progress(p.file.into(), v));
                ctx.request_repaint();
            });
            let _ = tx.send(match result {
                Ok(_) => Event::Done,
                Err(e) => Event::Error(format!("{e:#}")),
            });
            ctx.request_repaint();
        });
    }

    fn tag(&mut self, ctx: egui::Context, indices: Vec<usize>) {
        if self.worker.is_some() || indices.is_empty() {
            return;
        }
        let Some(data) = &self.data else { return };
        let jobs: Vec<_> = indices
            .into_iter()
            .filter_map(|i| data.item(i).map(|x| (i, x.image_path.clone())))
            .collect();
        let options = TaggerOptions {
            general_threshold: self.general,
            character_threshold: self.character,
            include_rating: self.rating,
        };
        let model = self.wd14_model;
        let use_legacy_cache = model == Wd14Model::default() && model_is_available(Self::cache());
        let (tx, rx) = mpsc::channel();
        self.worker = Some(rx);
        self.progress = 0.0;
        self.progress_text = model.label().to_owned();
        self.pending_predictions.clear();
        let cancel = Arc::new(AtomicBool::new(false));
        self.cancel = Some(cancel.clone());
        std::thread::spawn(move || {
            let report_progress = |p: crate::tagger::DownloadProgress| {
                let v = p
                    .total
                    .map(|t| p.downloaded as f32 / t as f32)
                    .unwrap_or(0.0);
                let _ = tx.send(Event::Progress(p.file.into(), v));
                ctx.request_repaint();
            };
            let tagger_result = if use_legacy_cache {
                Wd14Tagger::load_with_progress(Self::cache(), report_progress)
            } else {
                Wd14Tagger::load_model_with_progress(Self::cache(), model, report_progress)
            };
            let tagger = match tagger_result {
                Ok(t) => t,
                Err(e) => {
                    let _ = tx.send(Event::Error(format!("{e:#}")));
                    ctx.request_repaint();
                    return;
                }
            };
            let total = jobs.len();
            for (n, (_index, path)) in jobs.into_iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    let _ = tx.send(Event::Cancelled);
                    ctx.request_repaint();
                    return;
                }
                match tagger.predict_path(&path, options) {
                    Ok(p) => {
                        let _ = tx.send(Event::Predictions(path, p));
                    }
                    Err(e) => {
                        let _ = tx.send(Event::Error(format!("{e:#}")));
                        ctx.request_repaint();
                        return;
                    }
                }
                let _ = tx.send(Event::Progress(
                    model.label().into(),
                    (n + 1) as f32 / total as f32,
                ));
                ctx.request_repaint();
            }
            let _ = tx.send(Event::Done);
            ctx.request_repaint();
        });
    }

    fn poll(&mut self) {
        let Some(rx) = self.worker.take() else { return };
        let mut done = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                Event::Progress(s, v) => {
                    self.progress_text = s;
                    self.progress = v.clamp(0.0, 1.0);
                }
                Event::Predictions(i, p) => self.pending_predictions.push((i, p)),
                Event::Done => {
                    self.apply_pending_predictions();
                    self.status = "✓ WD14".into();
                    self.cancel = None;
                    done = true;
                }
                Event::Cancelled => {
                    self.pending_predictions.clear();
                    self.status = "× WD14".into();
                    self.cancel = None;
                    done = true;
                }
                Event::Error(e) => {
                    self.pending_predictions.clear();
                    self.status = format!("⚠ {e}");
                    self.cancel = None;
                    done = true;
                }
            }
        }
        if !done {
            self.worker = Some(rx);
        }
    }

    fn apply_pending_predictions(&mut self) {
        let banned: HashSet<_> = self
            .undesired
            .split(',')
            .map(|x| x.trim().to_ascii_lowercase())
            .filter(|x| !x.is_empty())
            .collect();
        let mut updates = Vec::with_capacity(self.pending_predictions.len());
        for (image_path, mut predictions) in self.pending_predictions.drain(..) {
            let Some(index) = self.data.as_ref().and_then(|data| {
                data.items
                    .iter()
                    .position(|item| item.image_path == image_path)
            }) else {
                continue;
            };
            if self.character_first {
                predictions.sort_by_key(|prediction| match prediction.category {
                    TagCategory::Character => 0,
                    TagCategory::General => 1,
                    TagCategory::Rating => 2,
                });
            }
            let mut tags: Vec<String> = predictions
                .into_iter()
                .map(|prediction| {
                    if self.spaces {
                        prediction.tag.replace('_', " ")
                    } else {
                        prediction.tag
                    }
                })
                .filter(|tag| {
                    let normalized = tag.to_ascii_lowercase();
                    !banned.contains(&normalized) && !banned.contains(&normalized.replace(' ', "_"))
                })
                .collect();
            if self.append {
                if let Some(item) = self.data.as_ref().and_then(|data| data.item(index)) {
                    let mut all = item.tags.clone();
                    all.append(&mut tags);
                    let mut seen = HashSet::new();
                    all.retain(|tag| seen.insert(tag.to_ascii_lowercase()));
                    tags = all;
                }
            }
            updates.push((index, tags.join(", ")));
        }
        if let Some(data) = &mut self.data {
            data.set_tag_texts(&updates);
        }
        self.caption_for = None;
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        let text_has_focus = ctx.wants_keyboard_input();
        let keys = ctx.input(|i| {
            (
                i.modifiers.ctrl && i.key_pressed(egui::Key::O),
                i.modifiers.ctrl && i.key_pressed(egui::Key::S),
                i.modifiers.ctrl && i.key_pressed(egui::Key::Z),
                i.modifiers.ctrl && i.key_pressed(egui::Key::Y),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Delete),
            )
        });
        if keys.0 {
            self.request_open();
        }
        if keys.1 {
            self.save();
        }
        if let Some(d) = &mut self.data {
            if keys.2 && !text_has_focus {
                d.undo();
                self.caption_for = None;
            }
            if keys.3 && !text_has_focus {
                d.redo();
                self.caption_for = None;
            }
            if keys.4 && !text_has_focus {
                if let Some(index) = self.current {
                    d.set_selected(index, true);
                }
            }
            if keys.5 && !text_has_focus {
                if let Some(index) = self.current {
                    d.set_selected(index, false);
                }
            }
        }
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("tools").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.visuals_mut().button_frame = false;
                if ui
                    .add_enabled(
                        self.worker.is_none(),
                        egui::Button::new(RichText::new("▣").size(21.0)),
                    )
                    .on_hover_text("Open · Ctrl+O")
                    .clicked()
                {
                    self.request_open();
                }
                if ui
                    .add_enabled(
                        self.data.is_some(),
                        egui::Button::new(RichText::new("◇").size(21.0)),
                    )
                    .on_hover_text("Save · Ctrl+S")
                    .clicked()
                {
                    self.save();
                }
                ui.separator();
                let undo = self.data.as_ref().map(Dataset::can_undo).unwrap_or(false);
                let redo = self.data.as_ref().map(Dataset::can_redo).unwrap_or(false);
                if ui
                    .add_enabled(undo, egui::Button::new("↶"))
                    .on_hover_text("Undo · Ctrl+Z")
                    .clicked()
                {
                    if let Some(d) = &mut self.data {
                        d.undo();
                        self.caption_for = None;
                    }
                }
                if ui
                    .add_enabled(redo, egui::Button::new("↷"))
                    .on_hover_text("Redo · Ctrl+Y")
                    .clicked()
                {
                    if let Some(d) = &mut self.data {
                        d.redo();
                        self.caption_for = None;
                    }
                }
                ui.separator();
                ui.add(
                    egui::TextEdit::singleline(&mut self.include)
                        .hint_text("＋ tag, tag")
                        .desired_width(145.0),
                )
                .on_hover_text("Include");
                ui.add(
                    egui::TextEdit::singleline(&mut self.exclude)
                        .hint_text("－ tag, tag")
                        .desired_width(145.0),
                )
                .on_hover_text("Exclude");
                if ui
                    .selectable_value(&mut self.mode, FilterMode::And, "AND")
                    .clicked()
                {
                    self.positive_filter_enabled = true;
                }
                if ui
                    .selectable_value(&mut self.mode, FilterMode::Or, "OR")
                    .clicked()
                {
                    self.positive_filter_enabled = true;
                }
                let positive = self.positive_picker.selected_tags_ref().len();
                let negative = self.negative_picker.selected_tags_ref().len();
                if positive + negative > 0 {
                    ui.label(
                        RichText::new(format!("+{positive} −{negative}"))
                            .small()
                            .color(Color32::from_rgb(120, 170, 230)),
                    );
                }
                ui.separator();
                let theme_icon = if self.dark_mode { "☀" } else { "☾" };
                if ui
                    .button(RichText::new(theme_icon).size(18.0))
                    .on_hover_text(if self.dark_mode {
                        "Light mode"
                    } else {
                        "Dark mode"
                    })
                    .clicked()
                {
                    self.dark_mode = !self.dark_mode;
                    apply_theme(ctx, self.dark_mode);
                }
                ui.label(RichText::new(&self.status).small().color(Color32::GRAY));
                if self.worker.is_some() {
                    ui.add(
                        egui::ProgressBar::new(self.progress)
                            .desired_width(110.0)
                            .text(&self.progress_text),
                    );
                }
            });
        });
    }

    fn side(&mut self, ctx: &egui::Context) {
        if ctx.available_rect().width() < COMPACT_LAYOUT_WIDTH {
            let response = egui::TopBottomPanel::bottom("side_compact")
                .resizable(true)
                .show_separator_line(false)
                .default_height(330.0)
                .min_height(210.0)
                .show(ctx, |ui| self.side_contents(ui));
            draw_pane_separator(ctx, response.response.rect, true, self.dark_mode);
        } else {
            let response = egui::SidePanel::right("side")
                .resizable(true)
                .show_separator_line(false)
                .default_width(390.0)
                .min_width(320.0)
                .max_width(520.0)
                .show(ctx, |ui| self.side_contents(ui));
            draw_pane_separator(ctx, response.response.rect, false, self.dark_mode);
        }
    }

    fn side_contents(&mut self, ui: &mut egui::Ui) {
        let previous_panel = self.panel;
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.panel, Panel::Edit, "✎")
                .on_hover_text("Caption");
            ui.selectable_value(&mut self.panel, Panel::Filter, "⌕")
                .on_hover_text("Tag filters");
            ui.selectable_value(&mut self.panel, Panel::Batch, "≋")
                .on_hover_text("Batch");
            ui.selectable_value(&mut self.panel, Panel::Wd14, "◉")
                .on_hover_text("WD14");
            ui.selectable_value(&mut self.panel, Panel::Files, "▤")
                .on_hover_text("Files / metadata");
        });
        if previous_panel == Panel::Edit && self.panel != Panel::Edit {
            self.commit_caption();
        }
        ui.separator();
        egui::ScrollArea::vertical()
            .id_source("side_contents_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| match self.panel {
                Panel::Edit => self.edit_ui(ui),
                Panel::Filter => self.filter_ui(ui),
                Panel::Batch => self.batch_ui(ui),
                Panel::Wd14 => self.wd_ui(ui),
                Panel::Files => self.files_ui(ui),
            });
    }

    fn edit_ui(&mut self, ui: &mut egui::Ui) {
        self.sync_caption();
        let Some(index) = self.current else {
            ui.centered_and_justified(|ui| {
                ui.label("—");
            });
            return;
        };
        if let Some(item) = self.data.as_ref().and_then(|d| d.item(index)) {
            ui.label(RichText::new(item.stem()).strong());
            ui.label(
                RichText::new(item.image_path.to_string_lossy())
                    .small()
                    .color(Color32::GRAY),
            )
            .on_hover_text(item.image_path.to_string_lossy());
        }
        let response = ui.add(
            egui::TextEdit::multiline(&mut self.caption)
                .hint_text("tag, tag")
                .desired_width(f32::INFINITY)
                .desired_rows(6)
                .lock_focus(true),
        );
        if response.lost_focus()
            || (response.has_focus()
                && ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Enter)))
        {
            if let Some(d) = &mut self.data {
                d.set_tag_text(index, &self.caption);
            }
        }
        ui.horizontal(|ui| {
            if ui
                .button("▶ 1")
                .on_hover_text("Apply to current image")
                .clicked()
            {
                if let Some(data) = &mut self.data {
                    data.set_tag_text(index, &self.caption);
                }
            }
            let visible = self.visible();
            if ui
                .add_enabled(
                    !visible.is_empty(),
                    egui::Button::new(format!("▶ ◫ {}", visible.len())),
                )
                .on_hover_text("Overwrite all visible captions")
                .clicked()
            {
                let updates: Vec<_> = visible
                    .into_iter()
                    .map(|item_index| (item_index, self.caption.clone()))
                    .collect();
                if let Some(data) = &mut self.data {
                    let changed = data.set_tag_texts(&updates);
                    self.status = format!("✓ {changed}");
                    self.caption_for = None;
                }
            }
        });

        let current_tags = self
            .data
            .as_ref()
            .and_then(|data| data.item(index))
            .map(|item| item.tags.clone())
            .unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("# {}", current_tags.len()))
                    .small()
                    .color(Color32::GRAY),
            );
            ui.label(RichText::new("click ±").small().color(Color32::DARK_GRAY))
                .on_hover_text("Click a tag to add or remove it from this caption");
        });

        let all = self.all_indices();
        let stats = self.tag_stats(&all);
        self.edit_picker
            .set_selected_tags(current_tags.iter().cloned());
        if self.edit_picker.show(ui, &stats).changed {
            let selected = self.edit_picker.selected_tags();
            let mut edited: Vec<String> = current_tags
                .into_iter()
                .filter(|tag| selected.contains(tag))
                .collect();
            for (tag, _) in &stats {
                if selected.contains(tag) && !edited.contains(tag) {
                    edited.push(tag.clone());
                }
            }
            if let Some(data) = &mut self.data {
                data.set_tags(index, edited);
                self.caption_for = None;
            }
        }
    }

    fn filter_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.filter_positive, true, "+")
                .on_hover_text("Positive filter");
            ui.selectable_value(&mut self.filter_positive, false, "−")
                .on_hover_text("Negative filter");
            ui.separator();
            let (enabled, mode) = if self.filter_positive {
                (&mut self.positive_filter_enabled, &mut self.mode)
            } else {
                (&mut self.negative_filter_enabled, &mut self.negative_mode)
            };
            if ui.selectable_label(!*enabled, "Ø").clicked() {
                *enabled = false;
            }
            if ui
                .selectable_label(*enabled && *mode == FilterMode::And, "AND")
                .clicked()
            {
                *enabled = true;
                *mode = FilterMode::And;
            }
            if ui
                .selectable_label(*enabled && *mode == FilterMode::Or, "OR")
                .clicked()
            {
                *enabled = true;
                *mode = FilterMode::Or;
            }
            if ui.button("×").on_hover_text("Clear this filter").clicked() {
                if self.filter_positive {
                    self.positive_picker.clear_selected_tags();
                    self.include.clear();
                } else {
                    self.negative_picker.clear_selected_tags();
                    self.exclude.clear();
                }
            }
            if ui.button("××").on_hover_text("Clear all filters").clicked() {
                self.positive_picker.clear_selected_tags();
                self.negative_picker.clear_selected_tags();
                self.include.clear();
                self.exclude.clear();
                self.selection_filter = false;
            }
        });

        let stats = self.tag_stats(&self.all_indices());
        if self.filter_positive {
            self.positive_picker.show(ui, &stats);
        } else {
            self.negative_picker.show(ui, &stats);
        }
        ui.separator();
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.selection_filter, "☑")
                .on_hover_text("Filter by checked images");
            let selected = self
                .data
                .as_ref()
                .map(|data| data.selected_indices().len())
                .unwrap_or(0);
            ui.label(
                RichText::new(selected.to_string())
                    .small()
                    .color(Color32::GRAY),
            );
        });
    }

    fn batch_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.batch_target, BatchTarget::Visible, "◫")
                .on_hover_text("Visible images");
            ui.selectable_value(&mut self.batch_target, BatchTarget::Checked, "☑")
                .on_hover_text("Checked images");
            ui.selectable_value(&mut self.batch_target, BatchTarget::Current, "1")
                .on_hover_text("Current image");
            let count = self.batch_indices().len();
            ui.label(
                RichText::new(format!("→ {count}"))
                    .small()
                    .color(Color32::GRAY),
            );
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.batch, Batch::Common, "◎")
                .on_hover_text("Edit common tags");
            ui.selectable_value(&mut self.batch, Batch::Append, "+")
                .on_hover_text("Append");
            ui.selectable_value(&mut self.batch, Batch::Prepend, "⇤")
                .on_hover_text("Prepend");
            ui.selectable_value(&mut self.batch, Batch::Remove, "−")
                .on_hover_text("Remove");
            ui.selectable_value(&mut self.batch, Batch::Replace, "↔")
                .on_hover_text("Replace");
            ui.selectable_value(&mut self.batch, Batch::Dedupe, "≠")
                .on_hover_text("Deduplicate");
            ui.selectable_value(&mut self.batch, Batch::Sort, "⇅")
                .on_hover_text("Sort tags");
            ui.selectable_value(&mut self.batch, Batch::Truncate, "⌁")
                .on_hover_text("Truncate");
        });

        let indices = self.batch_indices();
        let stats = self.tag_stats(&indices);
        match self.batch {
            Batch::Common => {
                let source = self
                    .data
                    .as_ref()
                    .map(|data| data.common_tags(&indices).join(", "))
                    .unwrap_or_default();
                if source != self.common_source {
                    self.common_source = source.clone();
                    self.common_edit = source;
                }
                ui.add(
                    egui::TextEdit::multiline(&mut self.common_edit)
                        .desired_width(f32::INFINITY)
                        .desired_rows(4)
                        .hint_text("common → edit"),
                )
                .on_hover_text(&self.common_source);
                ui.checkbox(&mut self.common_prepend, "⇤")
                    .on_hover_text("Prepend additional tags");
            }
            Batch::Append | Batch::Prepend => {
                ui.add(
                    egui::TextEdit::multiline(&mut self.batch_a)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3)
                        .hint_text("tag, tag"),
                );
            }
            Batch::Remove => {
                self.batch_picker.show(ui, &stats);
            }
            Batch::Replace => {
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.replace_target, ReplaceTarget::Selected, "☑")
                        .on_hover_text("Only selected tags");
                    ui.selectable_value(&mut self.replace_target, ReplaceTarget::Each, "#")
                        .on_hover_text("Each tag");
                    ui.selectable_value(&mut self.replace_target, ReplaceTarget::Caption, "¶")
                        .on_hover_text("Entire caption");
                    ui.checkbox(&mut self.regex, ".*").on_hover_text("Regex");
                });
                if self.replace_target == ReplaceTarget::Selected {
                    self.batch_picker.show(ui, &stats);
                }
                ui.add(egui::TextEdit::singleline(&mut self.batch_a).hint_text("from"));
                ui.add(egui::TextEdit::singleline(&mut self.batch_b).hint_text("to"));
            }
            Batch::Dedupe => {
                ui.label(RichText::new("A, A → A").color(Color32::GRAY));
            }
            Batch::Sort => {
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(&mut self.tag_sort_key, TagSortKey::Alphabetical, "A")
                        .on_hover_text("Alphabetical");
                    ui.selectable_value(&mut self.tag_sort_key, TagSortKey::Frequency, "#")
                        .on_hover_text("Frequency");
                    ui.selectable_value(&mut self.tag_sort_key, TagSortKey::Length, "↕")
                        .on_hover_text("Length");
                    ui.selectable_value(
                        &mut self.tag_sort_direction,
                        SortDirection::Ascending,
                        "↑",
                    );
                    ui.selectable_value(
                        &mut self.tag_sort_direction,
                        SortDirection::Descending,
                        "↓",
                    );
                });
            }
            Batch::Truncate => {
                ui.horizontal(|ui| {
                    ui.label("#");
                    ui.add(egui::DragValue::new(&mut self.truncate_count).clamp_range(0..=10_000));
                });
            }
        }

        let needs_selection = matches!(self.batch, Batch::Remove)
            || (self.batch == Batch::Replace && self.replace_target == ReplaceTarget::Selected);
        let enabled = !indices.is_empty()
            && (!needs_selection || !self.batch_picker.selected_tags_ref().is_empty());
        if ui
            .add_enabled(
                enabled,
                egui::Button::new(format!("▶ {}", indices.len()))
                    .min_size(egui::vec2(ui.available_width(), 36.0)),
            )
            .on_hover_text("Apply")
            .clicked()
        {
            let selected = self.batch_picker.selected_tags();
            let result: anyhow::Result<usize> = if let Some(data) = &mut self.data {
                match self.batch {
                    Batch::Common => {
                        let edited: Vec<String> = self
                            .common_edit
                            .split(',')
                            .map(|tag| tag.trim().to_owned())
                            .collect();
                        Ok(data.replace_common_tags(&indices, &edited, self.common_prepend))
                    }
                    Batch::Append => {
                        data.apply_operation(&indices, BatchOperation::Append(self.batch_a.clone()))
                    }
                    Batch::Prepend => data
                        .apply_operation(&indices, BatchOperation::Prepend(self.batch_a.clone())),
                    Batch::Remove => {
                        data.replace_selected_tags(&indices, selected, "^.*$", "", true)
                    }
                    Batch::Replace => match self.replace_target {
                        ReplaceTarget::Selected => data.replace_selected_tags(
                            &indices,
                            selected,
                            &self.batch_a,
                            &self.batch_b,
                            self.regex,
                        ),
                        ReplaceTarget::Each => data.apply_operation(
                            &indices,
                            if self.regex {
                                BatchOperation::Replace {
                                    pattern: self.batch_a.clone(),
                                    replacement: self.batch_b.clone(),
                                }
                            } else {
                                BatchOperation::ReplaceLiteral {
                                    from: self.batch_a.clone(),
                                    to: self.batch_b.clone(),
                                }
                            },
                        ),
                        ReplaceTarget::Caption => {
                            data.replace_caption(&indices, &self.batch_a, &self.batch_b, self.regex)
                        }
                    },
                    Batch::Dedupe => data.apply_operation(&indices, BatchOperation::Deduplicate),
                    Batch::Sort => Ok(data.sort_tags_in_items(
                        &indices,
                        self.tag_sort_key,
                        self.tag_sort_direction,
                    )),
                    Batch::Truncate => Ok(data.truncate_tags(&indices, self.truncate_count)),
                }
            } else {
                Ok(0)
            };
            match result {
                Ok(n) => {
                    self.status = format!("✓ {n}");
                    self.caption_for = None;
                    self.common_source.clear();
                }
                Err(error) => self.status = format!("⚠ {error}"),
            }
        }
    }

    fn files_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.worker.is_some();
        ui.checkbox(&mut self.backup_on_save, ".000")
            .on_hover_text("Create numbered caption backups when saving");
        ui.horizontal(|ui| {
            ui.label(".");
            ui.add(
                egui::TextEdit::singleline(&mut self.caption_extension)
                    .desired_width(70.0)
                    .hint_text("txt"),
            )
            .on_hover_text("Caption extension");
            ui.checkbox(&mut self.filename_fallback, "name")
                .on_hover_text("Use filename when caption is missing");
            let dirty = self
                .data
                .as_ref()
                .map(Dataset::has_unsaved_changes)
                .unwrap_or(false);
            if ui
                .add_enabled(!dirty && !busy, egui::Button::new("↻"))
                .on_hover_text("Reload captions")
                .clicked()
            {
                self.reload_dataset();
            }
        });
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.batch_target, BatchTarget::Visible, "◫")
                .on_hover_text("Visible images");
            ui.selectable_value(&mut self.batch_target, BatchTarget::Checked, "☑")
                .on_hover_text("Checked images");
            ui.selectable_value(&mut self.batch_target, BatchTarget::Current, "1")
                .on_hover_text("Current image");
            ui.label(
                RichText::new(format!("→ {}", self.batch_indices().len()))
                    .small()
                    .color(Color32::GRAY),
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.files_image, "▧")
                .on_hover_text("Image files");
            ui.checkbox(&mut self.files_caption, "¶")
                .on_hover_text("Caption files");
            ui.checkbox(&mut self.files_backups, "⋯")
                .on_hover_text("Caption backups");
        });
        ui.separator();
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.move_destination)
                    .desired_width(f32::INFINITY)
                    .hint_text("destination"),
            );
            if ui.button("▣").on_hover_text("Choose destination").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.move_destination = path.to_string_lossy().into_owned();
                }
            }
        });
        let dirty = self
            .data
            .as_ref()
            .map(Dataset::has_unsaved_changes)
            .unwrap_or(false);
        let target_count = self.batch_indices().len();
        let has_file_kind = self.files_image || self.files_caption || self.files_backups;
        if ui
            .add_enabled(
                !dirty
                    && !busy
                    && target_count > 0
                    && has_file_kind
                    && !self.move_destination.trim().is_empty(),
                egui::Button::new(format!("→ {target_count}"))
                    .min_size(egui::vec2(ui.available_width(), 34.0)),
            )
            .on_hover_text(if dirty {
                "Save changes first"
            } else {
                "Move files"
            })
            .clicked()
        {
            let destination = PathBuf::from(self.move_destination.trim());
            let groups: Vec<_> = self
                .batch_indices()
                .into_iter()
                .filter_map(|index| {
                    let item = self.data.as_ref()?.item(index)?;
                    Some((item.image_path.clone(), item.tag_path.clone()))
                })
                .collect();
            match move_file_groups(
                &groups,
                &destination,
                self.files_image,
                self.files_caption,
                self.files_backups,
            ) {
                Ok(moved) => self.status = format!("✓ {moved}"),
                Err(cause) => self.status = format!("⚠ {cause:#}"),
            }
            self.reload_dataset();
        }
        if ui
            .add_enabled(
                !dirty && !busy && target_count > 0 && has_file_kind,
                egui::Button::new(format!("⌫ {target_count}"))
                    .min_size(egui::vec2(ui.available_width(), 34.0)),
            )
            .on_hover_text(if dirty {
                "Save changes first"
            } else {
                "Delete files"
            })
            .clicked()
        {
            let files = self
                .batch_indices()
                .into_iter()
                .filter_map(|index| {
                    let item = self.data.as_ref()?.item(index)?;
                    Some((item.image_path.clone(), item.tag_path.clone()))
                })
                .collect();
            self.pending_delete = Some(DeletePlan {
                files,
                image: self.files_image,
                caption: self.files_caption,
                backups: self.files_backups,
            });
        }
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.metadata_mode, MetadataExportMode::Tags, "#")
                .on_hover_text("Export as tags");
            ui.selectable_value(&mut self.metadata_mode, MetadataExportMode::Caption, "¶")
                .on_hover_text("Export as caption");
            ui.separator();
            ui.selectable_value(&mut self.metadata_key, MetadataKeyMode::Stem, "stem")
                .on_hover_text("Stem keys");
            ui.selectable_value(
                &mut self.metadata_key,
                MetadataKeyMode::AbsolutePath,
                "path",
            )
            .on_hover_text("Absolute path keys");
            ui.separator();
            ui.selectable_value(&mut self.metadata_write, MetadataWriteMode::Merge, "+")
                .on_hover_text("Merge JSON");
            ui.selectable_value(&mut self.metadata_write, MetadataWriteMode::Overwrite, "↺")
                .on_hover_text("Overwrite JSON");
        });
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("↓ JSON"))
                .on_hover_text("Import kohya_ss metadata")
                .clicked()
            {
                self.import_metadata();
            }
            if ui
                .add_enabled(self.data.is_some(), egui::Button::new("↑ JSON"))
                .on_hover_text("Export kohya_ss metadata")
                .clicked()
            {
                self.export_metadata();
            }
            if ui.button("↺ cfg").on_hover_text("Reset settings").clicked() {
                self.apply_preferences(Preferences::default());
                apply_theme(ui.ctx(), self.dark_mode);
            }
        });
    }

    fn import_metadata(&mut self) {
        if self.worker.is_some() {
            self.status = "⚠ WD14 …".to_owned();
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file()
        else {
            return;
        };
        let document = match read_metadata(&path) {
            Ok(document) => document,
            Err(error) => {
                self.status = format!("⚠ {error:#}");
                return;
            }
        };
        let updates: Vec<(usize, String)> = self
            .data
            .as_ref()
            .map(|data| {
                data.items
                    .iter()
                    .enumerate()
                    .filter_map(|(index, item)| {
                        document
                            .tags_for_image(&item.image_path)
                            .map(|tags| (index, tags))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let Some(data) = &mut self.data {
            let changed = data.set_tag_texts(&updates);
            self.caption_for = None;
            self.status = format!("✓ JSON {changed}");
        }
    }

    fn export_metadata(&mut self) {
        self.commit_caption();
        let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name("metadata.json")
            .save_file()
        else {
            return;
        };
        let entries: Vec<(PathBuf, String)> = self
            .data
            .as_ref()
            .map(|data| {
                data.items
                    .iter()
                    .map(|item| (item.image_path.clone(), item.tag_text()))
                    .collect()
            })
            .unwrap_or_default();
        let options = MetadataExportOptions {
            mode: self.metadata_mode,
            write_mode: self.metadata_write,
            key_mode: self.metadata_key,
        };
        match write_metadata(&path, entries, options) {
            Ok(count) => self.status = format!("✓ JSON {count}"),
            Err(error) => self.status = format!("⚠ {error:#}"),
        }
    }

    fn reload_dataset(&mut self) {
        if self.worker.is_some() {
            self.status = "⚠ WD14 …".to_owned();
            return;
        }
        let Some(root) = self.data.as_ref().map(|data| data.root.clone()) else {
            return;
        };
        match Dataset::open_with_options(&root, &self.caption_extension, self.filename_fallback) {
            Ok(data) => {
                let count = data.len();
                self.data = Some(data);
                self.current = (count > 0).then_some(0);
                self.caption_for = None;
                self.textures.clear();
                self.texture_order.clear();
                self.broken.clear();
                self.loaded_caption_extension = self.caption_extension.clone();
            }
            Err(error) => self.status = format!("⚠ {error:#}"),
        }
    }

    fn delete_targets(&mut self, plan: DeletePlan) {
        if self.worker.is_some() {
            self.status = "⚠ WD14 …".to_owned();
            return;
        }
        match delete_file_groups(&plan.files, plan.image, plan.caption, plan.backups) {
            Ok(deleted) => self.status = format!("✓ {deleted}"),
            Err(error) => self.status = format!("⚠ {error:#}"),
        }
        self.reload_dataset();
    }

    fn wd_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.worker.is_some();
        ui.add_enabled_ui(!busy, |ui| {
            egui::ComboBox::from_id_source("wd14_model")
                .selected_text(self.wd14_model.label())
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    for model in Wd14Model::ALL {
                        ui.selectable_value(&mut self.wd14_model, model, model.label())
                            .on_hover_text(model.repository());
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("G");
            ui.add(egui::Slider::new(&mut self.general, 0.0..=1.0));
        });
        ui.horizontal(|ui| {
            ui.label("C");
            ui.add(egui::Slider::new(&mut self.character, 0.0..=1.0));
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.rating, "R").on_hover_text("Rating");
            ui.checkbox(&mut self.append, "+")
                .on_hover_text("Append / replace");
            ui.checkbox(&mut self.spaces, "_→␠")
                .on_hover_text("Underscore to space");
            ui.checkbox(&mut self.character_first, "C⇢")
                .on_hover_text("Character tags first");
        });
        ui.add(egui::TextEdit::singleline(&mut self.undesired).hint_text("− tag, tag"))
            .on_hover_text("Undesired tags");
        if busy {
            if let Some(cancel) = &self.cancel {
                if ui
                    .button("×")
                    .on_hover_text("Cancel after current image")
                    .clicked()
                {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        }
        let selected_model_available = model_is_available_for(Self::cache(), self.wd14_model)
            || (self.wd14_model == Wd14Model::default() && model_is_available(Self::cache()));
        if !selected_model_available
            && ui
                .add_enabled(
                    !busy,
                    egui::Button::new(format!("↓ {}", self.wd14_model.label())),
                )
                .on_hover_text(self.wd14_model.repository())
                .clicked()
        {
            self.download(ui.ctx().clone());
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy && self.current.is_some(), egui::Button::new("◉ 1"))
                .on_hover_text("Tag current")
                .clicked()
            {
                self.tag(ui.ctx().clone(), self.current.into_iter().collect());
            }
            let visible = self.visible();
            if ui
                .add_enabled(
                    !busy && !visible.is_empty(),
                    egui::Button::new(format!("◉ {}", visible.len())),
                )
                .on_hover_text("Tag visible")
                .clicked()
            {
                self.tag(ui.ctx().clone(), visible);
            }
        });
        ui.label(
            RichText::new(if self.append {
                "+ append"
            } else {
                "↺ replace"
            })
            .small()
            .color(Color32::GRAY),
        );
    }

    fn grid(&mut self, ctx: &egui::Context) {
        let visible = self.visible();
        let (dirty, selected) = self
            .data
            .as_ref()
            .map(|d| (d.modified_indices().len(), d.selected_indices().len()))
            .unwrap_or_default();
        egui::CentralPanel::default().show(ctx, |ui| {
            // Keep the virtualized grid's scrollbar separate from the pane
            // resize handle. The scrollbar only appears for larger datasets,
            // which otherwise makes the shared boundary flicker or look thick.
            ui.set_max_width((ui.available_width() - PANE_GUTTER).max(1.0));
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{} / {}",
                        visible.len(),
                        self.data.as_ref().map(Dataset::len).unwrap_or(0)
                    ))
                    .small(),
                );
                if dirty > 0 {
                    ui.label(
                        RichText::new(format!("● {dirty}"))
                            .small()
                            .color(Color32::YELLOW),
                    );
                }
                ui.separator();
                if ui
                    .small_button("☑")
                    .on_hover_text("Select visible")
                    .clicked()
                {
                    if let Some(d) = &mut self.data {
                        for &i in &visible {
                            d.set_selected(i, true);
                        }
                    }
                }
                if ui
                    .small_button("☐")
                    .on_hover_text("Clear selection")
                    .clicked()
                {
                    if let Some(d) = &mut self.data {
                        d.clear_selection();
                    }
                }
                if ui
                    .small_button("◩")
                    .on_hover_text("Invert visible selection")
                    .clicked()
                {
                    if let Some(d) = &mut self.data {
                        for &index in &visible {
                            d.toggle_selected(index);
                        }
                    }
                }
                ui.toggle_value(&mut self.selection_filter, "⌕☑")
                    .on_hover_text("Show checked images only");
                ui.label(
                    RichText::new(selected.to_string())
                        .small()
                        .color(Color32::GRAY),
                );
            });
            let columns = grid_column_count(ui.available_width(), ui.spacing().item_spacing.x);
            let rows = visible.len().div_ceil(columns);
            let row_height =
                thumbnail_card_height(ui.spacing().interact_size.y, ui.spacing().item_spacing.y);
            egui::ScrollArea::vertical()
                .id_source("image_grid_scroll")
                .auto_shrink([false, false])
                .show_rows(ui, row_height, rows, |ui, range| {
                    for row in range {
                        ui.horizontal(|ui| {
                            for col in 0..columns {
                                let p = row * columns + col;
                                if p >= visible.len() {
                                    break;
                                }
                                self.thumb(ui, visible[p]);
                            }
                        });
                    }
                });
        });
    }

    fn confirmations(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        if self.confirm_open {
            egui::Window::new("●")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label("Unsaved");
                    ui.horizontal(|ui| {
                        if ui.button("◇ → ▣").on_hover_text("Save and open").clicked() {
                            self.save();
                            if !self
                                .data
                                .as_ref()
                                .map(Dataset::has_unsaved_changes)
                                .unwrap_or(false)
                            {
                                self.confirm_open = false;
                                self.open();
                            }
                        }
                        if ui.button("▣").on_hover_text("Discard and open").clicked() {
                            self.confirm_open = false;
                            self.open();
                        }
                        if ui.button("×").on_hover_text("Cancel").clicked() {
                            self.confirm_open = false;
                        }
                    });
                });
        }
        if self.confirm_exit {
            egui::Window::new("● ")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label("Unsaved");
                    ui.horizontal(|ui| {
                        if ui.button("◇ → ×").on_hover_text("Save and exit").clicked() {
                            self.save();
                            if !self
                                .data
                                .as_ref()
                                .map(Dataset::has_unsaved_changes)
                                .unwrap_or(false)
                            {
                                self.allow_exit = true;
                                frame.close();
                            }
                        }
                        if ui.button("×").on_hover_text("Discard and exit").clicked() {
                            self.allow_exit = true;
                            frame.close();
                        }
                        if ui.button("↶").on_hover_text("Cancel").clicked() {
                            self.confirm_exit = false;
                        }
                    });
                });
        }
        if let Some(count) = self.pending_delete.as_ref().map(|plan| plan.files.len()) {
            egui::Window::new("⌫")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(format!("{count} files"));
                    ui.horizontal(|ui| {
                        if ui.button("⌫").on_hover_text("Delete permanently").clicked() {
                            if let Some(plan) = self.pending_delete.take() {
                                self.delete_targets(plan);
                            }
                        }
                        if ui.button("↶").on_hover_text("Cancel").clicked() {
                            self.pending_delete = None;
                        }
                    });
                });
        }
    }

    fn thumb(&mut self, ui: &mut egui::Ui, index: usize) {
        let Some(item) = self.data.as_ref().and_then(|d| d.item(index)).cloned() else {
            return;
        };
        let texture = self.texture(ui.ctx(), &item.image_path);
        let selected = self.current == Some(index);
        let dark_mode = ui.visuals().dark_mode;
        let card_fill = match (dark_mode, selected) {
            (true, true) => Color32::from_rgb(42, 68, 96),
            (true, false) => Color32::from_rgb(28, 30, 34),
            (false, true) => Color32::from_rgb(214, 230, 250),
            (false, false) => Color32::from_rgb(248, 249, 251),
        };
        let card_stroke = match (dark_mode, selected) {
            (_, true) => egui::Stroke::new(1.5_f32, Color32::from_rgb(58, 126, 218)),
            (true, false) => egui::Stroke::new(1.0_f32, Color32::from_rgb(42, 45, 52)),
            (false, false) => egui::Stroke::new(1.0_f32, Color32::from_rgb(205, 210, 218)),
        };
        egui::Frame::none()
            .fill(card_fill)
            .rounding(6.0)
            .stroke(card_stroke)
            .inner_margin(5.0)
            .show(ui, |ui| {
                ui.set_width(THUMB);
                ui.vertical(|ui| {
                    let clicked = if let Some(t) = texture {
                        let size = t.size_vec2();
                        let scale = (THUMB / size.x).min(THUMB_IMAGE_HEIGHT / size.y);
                        let (image_slot, response) = ui.allocate_exact_size(
                            egui::vec2(THUMB, THUMB_IMAGE_HEIGHT),
                            egui::Sense::click(),
                        );
                        let image_rect = egui::Rect::from_center_size(
                            image_slot.center(),
                            size * scale,
                        );
                        ui.painter().image(
                            t.id(),
                            image_rect,
                            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                        response.clicked()
                    } else {
                        ui.add_sized(
                            [THUMB, THUMB_IMAGE_HEIGHT],
                            egui::Button::new("⚠").frame(false),
                        )
                            .clicked()
                    };
                    if clicked {
                        self.commit_caption();
                        self.current = Some(index);
                        self.caption_for = None;
                    }
                    ui.horizontal(|ui| {
                        let mut checked = item.selected;
                        if ui.checkbox(&mut checked, "").changed() {
                            if let Some(d) = &mut self.data {
                                d.set_selected(index, checked);
                            }
                        }
                        let stem: String = item.stem().chars().take(18).collect();
                        ui.label(
                            RichText::new(stem)
                                .small()
                                .color(if item.is_modified() {
                                    if dark_mode {
                                        Color32::YELLOW
                                    } else {
                                        Color32::from_rgb(170, 105, 0)
                                    }
                                } else if dark_mode {
                                    Color32::LIGHT_GRAY
                                } else {
                                    Color32::from_rgb(48, 52, 58)
                                }),
                        )
                        .on_hover_text(item.tag_text());
                    });
                    let (_, tags_rect) =
                        ui.allocate_space(egui::vec2(THUMB, THUMB_TAGS_HEIGHT));
                    ui.allocate_ui_at_rect(tags_rect, |ui| {
                        ui.set_clip_rect(ui.clip_rect().intersect(tags_rect));
                        ui.horizontal_wrapped(|ui| {
                        for tag in item.tags.iter().take(3) {
                            let selected = self.batch_picker.selected_tags_ref().contains(tag);
                            let shown: String = tag.chars().take(15).collect();
                            let response = ui
                                .selectable_label(selected, RichText::new(shown).size(10.0))
                                .on_hover_text(format!(
                                    "{tag}\nclick: batch · Ctrl: +filter · Alt: −filter · Shift: remove"
                                ));
                            if response.clicked() {
                                let modifiers = ui.input(|input| input.modifiers);
                                if modifiers.shift {
                                    if let Some(data) = &mut self.data {
                                        data.remove_tag(index, tag);
                                        self.caption_for = None;
                                    }
                                } else if modifiers.ctrl {
                                    let mut tags = self.positive_picker.selected_tags();
                                    if !tags.remove(tag) {
                                        tags.insert(tag.clone());
                                    }
                                    self.positive_picker.set_selected_tags(tags);
                                } else if modifiers.alt {
                                    let mut tags = self.negative_picker.selected_tags();
                                    if !tags.remove(tag) {
                                        tags.insert(tag.clone());
                                    }
                                    self.negative_picker.set_selected_tags(tags);
                                } else {
                                    let mut tags = self.batch_picker.selected_tags();
                                    if !tags.remove(tag) {
                                        tags.insert(tag.clone());
                                    }
                                    self.batch_picker.set_selected_tags(tags);
                                }
                            }
                        }
                        if item.tags.len() > 3 {
                            ui.label(
                                RichText::new(format!("+{}", item.tags.len() - 3))
                                    .size(10.0)
                                    .color(Color32::GRAY),
                            );
                        }
                        });
                    });
                });
            });
    }
}

fn grid_column_count(available_width: f32, gap: f32) -> usize {
    (((available_width + gap) / (THUMB_CARD_WIDTH + gap)).floor() as usize).max(1)
}

fn thumbnail_card_height(interact_height: f32, vertical_gap: f32) -> f32 {
    THUMB_IMAGE_HEIGHT
        + interact_height
        + THUMB_TAGS_HEIGHT
        + vertical_gap * 2.0
        + THUMB_CARD_MARGIN
}

fn draw_pane_separator(
    ctx: &egui::Context,
    panel_rect: egui::Rect,
    horizontal: bool,
    dark_mode: bool,
) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("pane_separator"),
    ));
    let stroke = egui::Stroke::new(
        1.0_f32,
        if dark_mode {
            Color32::from_rgb(58, 62, 70)
        } else {
            Color32::from_rgb(190, 196, 206)
        },
    );
    if horizontal {
        let y = painter.round_to_pixel(panel_rect.top() + 1.0);
        painter.hline(panel_rect.x_range(), y, stroke);
    } else {
        let x = painter.round_to_pixel(panel_rect.left() + 1.0);
        painter.vline(x, panel_rect.y_range(), stroke);
    }
}

fn apply_theme(ctx: &egui::Context, dark_mode: bool) {
    let accent = Color32::from_rgb(58, 126, 218);
    let mut visuals = if dark_mode {
        let mut visuals = egui::Visuals::dark();
        visuals.widgets.active.bg_fill = Color32::from_rgb(48, 96, 158);
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(48, 58, 72);
        visuals.panel_fill = Color32::from_rgb(20, 22, 26);
        visuals.window_fill = Color32::from_rgb(24, 26, 31);
        visuals
    } else {
        let mut visuals = egui::Visuals::light();
        visuals.widgets.active.bg_fill = Color32::from_rgb(190, 215, 247);
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(226, 236, 249);
        visuals.panel_fill = Color32::from_rgb(244, 246, 249);
        visuals.window_fill = Color32::from_rgb(250, 251, 253);
        visuals.faint_bg_color = Color32::from_rgb(238, 241, 246);
        visuals
    };
    visuals.selection.bg_fill = accent;
    visuals.selection.stroke.color = Color32::WHITE;
    ctx.set_visuals(visuals);
    ctx.request_repaint();
}

fn configure_fonts(ctx: &egui::Context) {
    let Some(windows) = std::env::var_os("WINDIR").map(PathBuf::from) else {
        return;
    };
    let mut fonts = egui::FontDefinitions::default();
    let candidates = [
        ("segoe-ui", windows.join("Fonts").join("segoeui.ttf")),
        ("segoe-symbol", windows.join("Fonts").join("seguisym.ttf")),
    ];
    let mut loaded = Vec::new();
    for (name, path) in candidates {
        if let Ok(bytes) = std::fs::read(path) {
            fonts
                .font_data
                .insert(name.to_owned(), egui::FontData::from_owned(bytes));
            loaded.push(name.to_owned());
        }
    }
    if loaded.is_empty() {
        return;
    }
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        if let Some(names) = fonts.families.get_mut(&family) {
            for name in loaded.iter().rev() {
                names.insert(0, name.clone());
            }
        }
    }
    ctx.set_fonts(fonts);
}

impl eframe::App for TaggerNeoApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.poll();
        self.shortcuts(ctx);
        self.toolbar(ctx);
        self.side(ctx);
        self.grid(ctx);
        self.confirmations(ctx, frame);
    }

    fn on_close_event(&mut self) -> bool {
        self.commit_caption();
        if self.allow_exit
            || !self
                .data
                .as_ref()
                .map(Dataset::has_unsaved_changes)
                .unwrap_or(false)
        {
            true
        } else {
            self.confirm_exit = true;
            false
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, PREFS_KEY, &self.preferences());
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn grid_columns_include_card_margins_and_gaps() {
        assert_eq!(grid_column_count(159.0, 8.0), 1);
        assert_eq!(grid_column_count(319.0, 8.0), 1);
        assert_eq!(grid_column_count(328.0, 8.0), 2);
    }

    #[test]
    fn virtualized_thumbnail_rows_have_a_fixed_height() {
        assert_eq!(thumbnail_card_height(18.0, 7.0), 226.0);
        assert_eq!(thumbnail_card_height(18.0, 7.0) * 10_000.0, 2_260_000.0);
    }

    #[test]
    fn theme_switch_updates_egui_visuals() {
        let context = egui::Context::default();
        apply_theme(&context, false);
        assert!(!context.style().visuals.dark_mode);
        apply_theme(&context, true);
        assert!(context.style().visuals.dark_mode);
    }

    #[test]
    fn preferences_default_and_round_trip_wd14_model() {
        let old_preferences: Preferences = serde_json::from_str("{}").unwrap();
        assert_eq!(old_preferences.wd14_model, Wd14Model::default());

        let preferences = Preferences {
            wd14_model: Wd14Model::Eva02LargeV3,
            ..Preferences::default()
        };
        let restored: Preferences =
            serde_json::from_str(&serde_json::to_string(&preferences).unwrap()).unwrap();
        assert_eq!(restored.wd14_model, Wd14Model::Eva02LargeV3);
    }
}
