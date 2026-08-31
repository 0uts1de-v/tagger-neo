//! WD14 model management, image preparation, and ONNX inference.

use anyhow::{bail, Context, Result};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
use ort::value::{Tensor, TensorElementType};
use ort::{ep, session::Session};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const MODEL_REPOSITORY: &str = "SmilingWolf/wd-v1-4-convnext-tagger-v2";
pub const DEFAULT_IMAGE_SIZE: usize = 448;
const MODEL_REVISION: &str = "main";
const MODEL_FILE: &str = "model.onnx";
const LABEL_FILE: &str = "selected_tags.csv";
const MIN_MODEL_BYTES: u64 = 1_000_000;
const MIN_LABEL_BYTES: u64 = 1_000;

/// WD14 models offered by kohya_ss GUI.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Wd14Model {
    #[default]
    #[serde(rename = "wd-v1-4-convnext-tagger-v2")]
    ConvNextV2,
    #[serde(rename = "wd-v1-4-convnextv2-tagger-v2")]
    ConvNextV2V2,
    #[serde(rename = "wd-v1-4-vit-tagger-v2")]
    VitV2,
    #[serde(rename = "wd-v1-4-swinv2-tagger-v2")]
    SwinV2V2,
    #[serde(rename = "wd-v1-4-moat-tagger-v2")]
    MoatV2,
    #[serde(rename = "wd-swinv2-tagger-v3")]
    SwinV2V3,
    #[serde(rename = "wd-vit-tagger-v3")]
    VitV3,
    #[serde(rename = "wd-convnext-tagger-v3")]
    ConvNextV3,
    #[serde(rename = "wd-eva02-large-tagger-v3")]
    Eva02LargeV3,
}

impl Wd14Model {
    pub const ALL: [Self; 9] = [
        Self::ConvNextV2,
        Self::ConvNextV2V2,
        Self::VitV2,
        Self::SwinV2V2,
        Self::MoatV2,
        Self::SwinV2V3,
        Self::VitV3,
        Self::ConvNextV3,
        Self::Eva02LargeV3,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ConvNextV2 => "ConvNeXt v2",
            Self::ConvNextV2V2 => "ConvNeXtV2 v2",
            Self::VitV2 => "ViT v2",
            Self::SwinV2V2 => "SwinV2 v2",
            Self::MoatV2 => "MOAT v2",
            Self::SwinV2V3 => "SwinV2 v3",
            Self::VitV3 => "ViT v3",
            Self::ConvNextV3 => "ConvNeXt v3",
            Self::Eva02LargeV3 => "EVA02-Large v3",
        }
    }

    pub const fn repository(self) -> &'static str {
        match self {
            Self::ConvNextV2 => "SmilingWolf/wd-v1-4-convnext-tagger-v2",
            Self::ConvNextV2V2 => "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
            Self::VitV2 => "SmilingWolf/wd-v1-4-vit-tagger-v2",
            Self::SwinV2V2 => "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
            Self::MoatV2 => "SmilingWolf/wd-v1-4-moat-tagger-v2",
            Self::SwinV2V3 => "SmilingWolf/wd-swinv2-tagger-v3",
            Self::VitV3 => "SmilingWolf/wd-vit-tagger-v3",
            Self::ConvNextV3 => "SmilingWolf/wd-convnext-tagger-v3",
            Self::Eva02LargeV3 => "SmilingWolf/wd-eva02-large-tagger-v3",
        }
    }

    pub const fn cache_key(self) -> &'static str {
        match self {
            Self::ConvNextV2 => "wd-v1-4-convnext-tagger-v2",
            Self::ConvNextV2V2 => "wd-v1-4-convnextv2-tagger-v2",
            Self::VitV2 => "wd-v1-4-vit-tagger-v2",
            Self::SwinV2V2 => "wd-v1-4-swinv2-tagger-v2",
            Self::MoatV2 => "wd-v1-4-moat-tagger-v2",
            Self::SwinV2V3 => "wd-swinv2-tagger-v3",
            Self::VitV3 => "wd-vit-tagger-v3",
            Self::ConvNextV3 => "wd-convnext-tagger-v3",
            Self::Eva02LargeV3 => "wd-eva02-large-tagger-v3",
        }
    }

    pub fn from_id(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|model| {
            value == model.repository() || value == model.cache_key() || value == model.label()
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TagCategory {
    Rating,
    General,
    Character,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TagPrediction {
    pub tag: String,
    pub confidence: f32,
    pub category: TagCategory,
}

#[derive(Clone, Copy, Debug)]
pub struct TaggerOptions {
    pub general_threshold: f32,
    pub character_threshold: f32,
    pub include_rating: bool,
}

impl Default for TaggerOptions {
    fn default() -> Self {
        Self {
            general_threshold: 0.35,
            character_threshold: 0.35,
            include_rating: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelFiles {
    pub model: PathBuf,
    pub labels: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub struct DownloadProgress {
    pub file: &'static str,
    pub downloaded: u64,
    pub total: Option<u64>,
}

#[derive(Clone, Debug)]
struct Label {
    name: String,
    category: TagCategory,
}

/// Synchronous tagger. A GUI should construct and call it on a worker thread.
pub struct Wd14Tagger {
    model: Mutex<Session>,
    labels: Vec<Label>,
    image_size: usize,
}

impl Wd14Tagger {
    /// Loads the historical default model from a legacy, flat cache directory.
    pub fn load(cache_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_progress(cache_dir, |_| {})
    }

    pub fn load_with_progress(
        cache_dir: impl AsRef<Path>,
        mut progress: impl FnMut(DownloadProgress),
    ) -> Result<Self> {
        let files = ensure_model_files(cache_dir.as_ref(), &mut progress)?;
        Self::load_files(files)
    }

    /// Loads a selected model from its own directory below `cache_root`.
    pub fn load_model(cache_root: impl AsRef<Path>, model: Wd14Model) -> Result<Self> {
        Self::load_model_with_progress(cache_root, model, |_| {})
    }

    pub fn load_model_with_progress(
        cache_root: impl AsRef<Path>,
        model: Wd14Model,
        mut progress: impl FnMut(DownloadProgress),
    ) -> Result<Self> {
        let files = ensure_model_files_for(cache_root, model, &mut progress)?;
        Self::load_files(files)
    }

    fn load_files(files: ModelFiles) -> Result<Self> {
        let labels = read_labels(&files.labels)?;
        // ONNX Runtime validates and optimizes the graph while creating the session.
        let builder = Session::builder().context("could not initialize ONNX Runtime")?;
        let builder = builder.with_memory_pattern(false).map_err(|error| {
            anyhow::anyhow!("could not configure DirectML memory handling: {error}")
        })?;
        let model = builder
            .with_execution_providers([ep::DirectML::default().build().error_on_failure()])
            .map_err(|error| {
                anyhow::anyhow!("could not initialize the DirectML execution provider: {error}")
            })?
            .commit_from_file(&files.model)
            .with_context(|| format!("invalid ONNX model: {}", files.model.display()))?;
        let input = model.inputs().first().context("WD14 model has no input")?;
        let shape = input
            .dtype()
            .tensor_shape()
            .context("WD14 input is not a tensor")?;
        if input.dtype().tensor_type() != Some(TensorElementType::Float32) || shape.len() != 4 {
            bail!("unexpected WD14 input tensor: {:?}", input.dtype());
        }
        let height = usize::try_from(shape[1]).context("WD14 input height is not fixed")?;
        let width = usize::try_from(shape[2]).context("WD14 input width is not fixed")?;
        let channels =
            usize::try_from(shape[3]).context("WD14 input channel count is not fixed")?;
        if height != width || channels != 3 {
            bail!("unexpected WD14 input shape: {:?}", shape);
        }
        Ok(Self {
            model: Mutex::new(model),
            labels,
            image_size: height,
        })
    }

    pub fn image_size(&self) -> usize {
        self.image_size
    }

    pub fn predict_path(
        &self,
        image_path: impl AsRef<Path>,
        options: TaggerOptions,
    ) -> Result<Vec<TagPrediction>> {
        let image = image::open(image_path.as_ref())
            .with_context(|| format!("could not open image: {}", image_path.as_ref().display()))?;
        self.predict_image(&image, options)
    }

    pub fn predict_image(
        &self,
        image: &DynamicImage,
        options: TaggerOptions,
    ) -> Result<Vec<TagPrediction>> {
        validate_options(options)?;
        let pixels = preprocess_image(image, self.image_size)?;
        let input = Tensor::from_array((
            [1usize, self.image_size, self.image_size, 3],
            pixels.into_boxed_slice(),
        ))
        .context("could not create WD14 input tensor")?;
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("WD14 model lock was poisoned"))?;
        let outputs = model
            .run(ort::inputs![input])
            .context("WD14 inference failed")?;
        let output = outputs
            .values()
            .next()
            .context("WD14 model returned no output")?;
        let (_, probabilities) = output
            .try_extract_tensor::<f32>()
            .context("WD14 output is not a float32 tensor")?;
        select_predictions(&self.labels, probabilities, options)
    }
}

pub fn model_files(cache_dir: impl AsRef<Path>) -> ModelFiles {
    ModelFiles {
        model: cache_dir.as_ref().join(MODEL_FILE),
        labels: cache_dir.as_ref().join(LABEL_FILE),
    }
}

pub fn model_files_for(cache_root: impl AsRef<Path>, model: Wd14Model) -> ModelFiles {
    model_files(cache_root.as_ref().join(model.cache_key()))
}

pub fn model_is_available(cache_dir: impl AsRef<Path>) -> bool {
    let files = model_files(cache_dir);
    model_file_is_plausible(&files.model) && labels_file_is_plausible(&files.labels)
}

pub fn model_is_available_for(cache_root: impl AsRef<Path>, model: Wd14Model) -> bool {
    let files = model_files_for(cache_root, model);
    model_file_is_plausible(&files.model) && labels_file_is_plausible(&files.labels)
}

pub fn ensure_model_files(
    cache_dir: impl AsRef<Path>,
    progress: &mut impl FnMut(DownloadProgress),
) -> Result<ModelFiles> {
    let cache_dir = cache_dir.as_ref();
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("could not create model cache: {}", cache_dir.display()))?;
    let files = model_files(cache_dir);
    if !model_file_is_plausible(&files.model) {
        download_file(
            MODEL_REPOSITORY,
            MODEL_FILE,
            &files.model,
            MIN_MODEL_BYTES,
            progress,
        )?;
    }
    if !labels_file_is_plausible(&files.labels) {
        download_file(
            MODEL_REPOSITORY,
            LABEL_FILE,
            &files.labels,
            MIN_LABEL_BYTES,
            progress,
        )?;
        if let Err(error) = read_labels(&files.labels) {
            let _ = fs::remove_file(&files.labels);
            return Err(error).context("downloaded label file is invalid");
        }
    }
    Ok(files)
}

pub fn ensure_model_files_for(
    cache_root: impl AsRef<Path>,
    model: Wd14Model,
    progress: &mut impl FnMut(DownloadProgress),
) -> Result<ModelFiles> {
    let files = model_files_for(cache_root, model);
    let cache_dir = files
        .model
        .parent()
        .context("model cache path has no parent")?;
    fs::create_dir_all(cache_dir)
        .with_context(|| format!("could not create model cache: {}", cache_dir.display()))?;
    if !model_file_is_plausible(&files.model) {
        download_file(
            model.repository(),
            MODEL_FILE,
            &files.model,
            MIN_MODEL_BYTES,
            progress,
        )?;
    }
    if !labels_file_is_plausible(&files.labels) {
        download_file(
            model.repository(),
            LABEL_FILE,
            &files.labels,
            MIN_LABEL_BYTES,
            progress,
        )?;
        if let Err(error) = read_labels(&files.labels) {
            let _ = fs::remove_file(&files.labels);
            return Err(error).context("downloaded label file is invalid");
        }
    }
    Ok(files)
}

fn download_file(
    repository: &str,
    name: &'static str,
    target: &Path,
    minimum_size: u64,
    progress: &mut impl FnMut(DownloadProgress),
) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}?download=true",
        repository, MODEL_REVISION, name
    );
    let part = target.with_extension(format!(
        "{}.part",
        target
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("download")
    ));
    if part.exists() {
        fs::remove_file(&part).with_context(|| {
            format!(
                "could not remove stale partial download: {}",
                part.display()
            )
        })?;
    }
    let result = (|| -> Result<()> {
        let mut response = reqwest::blocking::Client::builder()
            .user_agent(concat!("tagger-neo/", env!("CARGO_PKG_VERSION")))
            .build()?
            .get(&url)
            .send()
            .with_context(|| format!("request failed for {name}"))?
            .error_for_status()
            .with_context(|| format!("server rejected download for {name}"))?;
        let http_total = response.content_length();
        let total = http_total;
        let mut writer = BufWriter::new(
            File::create(&part).with_context(|| format!("could not create {}", part.display()))?,
        );
        let mut downloaded = 0_u64;
        let mut buffer = [0_u8; 128 * 1024];
        progress(DownloadProgress {
            file: name,
            downloaded,
            total,
        });
        loop {
            let count = response
                .read(&mut buffer)
                .context("download stream failed")?;
            if count == 0 {
                break;
            }
            writer
                .write_all(&buffer[..count])
                .context("could not write download")?;
            downloaded += count as u64;
            progress(DownloadProgress {
                file: name,
                downloaded,
                total,
            });
        }
        writer.flush().context("could not flush download")?;
        drop(writer);
        if let Some(advertised) = http_total {
            if downloaded != advertised {
                bail!(
                    "incomplete {name}: downloaded {downloaded} bytes, server advertised {advertised}"
                );
            }
        }
        if downloaded < minimum_size {
            bail!("incomplete {name}: only {downloaded} bytes received");
        }
        if target.exists() {
            fs::remove_file(target)
                .with_context(|| format!("could not replace {}", target.display()))?;
        }
        fs::rename(&part, target).with_context(|| {
            format!(
                "could not install {} -> {}",
                part.display(),
                target.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&part);
    }
    result
}

fn file_has_minimum_size(path: &Path, minimum: u64) -> bool {
    path.metadata()
        .map(|m| m.is_file() && m.len() >= minimum)
        .unwrap_or(false)
}

fn model_file_is_plausible(path: &Path) -> bool {
    file_has_minimum_size(path, MIN_MODEL_BYTES)
}

fn labels_file_is_plausible(path: &Path) -> bool {
    path.metadata()
        .map(|m| m.is_file() && m.len() >= MIN_LABEL_BYTES)
        .unwrap_or(false)
        && read_labels(path).is_ok()
}

fn read_labels(path: &Path) -> Result<Vec<Label>> {
    let file =
        File::open(path).with_context(|| format!("could not open labels: {}", path.display()))?;
    parse_labels(BufReader::new(file))
}

fn parse_labels(reader: impl Read) -> Result<Vec<Label>> {
    let mut csv = csv::Reader::from_reader(reader);
    let header = csv.headers().context("could not read label CSV header")?;
    if header.get(0) != Some("tag_id")
        || header.get(1) != Some("name")
        || header.get(2) != Some("category")
    {
        bail!("unexpected label CSV header");
    }
    let mut labels = Vec::new();
    for (line, row) in csv.records().enumerate() {
        let row = row.with_context(|| format!("invalid label CSV row {}", line + 2))?;
        let name = row.get(1).context("label has no name")?.trim();
        if name.is_empty() {
            bail!("empty label name at CSV row {}", line + 2);
        }
        let category = match row.get(2) {
            Some("9") => TagCategory::Rating,
            Some("0") => TagCategory::General,
            Some("4") => TagCategory::Character,
            Some(value) => bail!("unsupported label category {value} at CSV row {}", line + 2),
            None => bail!("label has no category at CSV row {}", line + 2),
        };
        labels.push(Label {
            name: name.to_owned(),
            category,
        });
    }
    if labels.is_empty() {
        bail!("label CSV is empty");
    }
    Ok(labels)
}

fn validate_options(options: TaggerOptions) -> Result<()> {
    for (name, value) in [
        ("general threshold", options.general_threshold),
        ("character threshold", options.character_threshold),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            bail!("{name} must be between 0 and 1");
        }
    }
    Ok(())
}

fn select_predictions(
    labels: &[Label],
    probabilities: &[f32],
    options: TaggerOptions,
) -> Result<Vec<TagPrediction>> {
    validate_options(options)?;
    if labels.len() != probabilities.len() {
        bail!(
            "model output/label count mismatch: {} probabilities, {} labels",
            probabilities.len(),
            labels.len()
        );
    }
    // kohya_ss treats ratings as mutually exclusive and selects their argmax.
    let best_rating = labels
        .iter()
        .zip(probabilities)
        .enumerate()
        .filter(|(_, (label, score))| label.category == TagCategory::Rating && score.is_finite())
        .max_by(|(_, (_, a)), (_, (_, b))| a.total_cmp(b))
        .map(|(index, _)| index);
    let mut predictions = Vec::new();
    for (index, (label, &confidence)) in labels.iter().zip(probabilities).enumerate() {
        if !confidence.is_finite() {
            continue;
        }
        let selected = match label.category {
            TagCategory::Rating => options.include_rating && best_rating == Some(index),
            TagCategory::General => confidence >= options.general_threshold,
            TagCategory::Character => confidence >= options.character_threshold,
        };
        if selected {
            predictions.push(TagPrediction {
                tag: label.name.clone(),
                confidence,
                category: label.category,
            });
        }
    }
    predictions.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    Ok(predictions)
}

/// White-composite alpha, centre-pad, resize, then emit unnormalised BGR/HWC.
fn preprocess_image(image: &DynamicImage, size: usize) -> Result<Vec<f32>> {
    if size == 0 {
        bail!("model image size must be non-zero");
    }
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();
    if width == 0 || height == 0 {
        bail!("image has zero width or height");
    }
    let side = width.max(height);
    let left = (side - width) / 2;
    let top = (side - height) / 2;
    let mut square: RgbImage = ImageBuffer::from_pixel(side, side, Rgb([255, 255, 255]));
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = pixel[3] as u32;
        let composite = |channel: u8| -> u8 {
            ((channel as u32 * alpha + 255 * (255 - alpha) + 127) / 255) as u8
        };
        square.put_pixel(
            x + left,
            y + top,
            Rgb([
                composite(pixel[0]),
                composite(pixel[1]),
                composite(pixel[2]),
            ]),
        );
    }
    let resized = image::imageops::resize(
        &square,
        size as u32,
        size as u32,
        // Catmull-Rom is the image crate's bicubic filter and matches the
        // reference WD14 preprocessing more closely than Lanczos.
        image::imageops::FilterType::CatmullRom,
    );
    let mut bgr = Vec::with_capacity(size * size * 3);
    for pixel in resized.pixels() {
        bgr.push(pixel[2] as f32);
        bgr.push(pixel[1] as f32);
        bgr.push(pixel[0] as f32);
    }
    Ok(bgr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgba, RgbaImage};
    use std::collections::HashSet;
    use std::io::Cursor;

    #[test]
    fn model_catalog_matches_kohya_ss_and_has_unique_caches() {
        let repositories = Wd14Model::ALL
            .iter()
            .map(|model| model.repository())
            .collect::<Vec<_>>();
        assert_eq!(
            repositories,
            vec![
                "SmilingWolf/wd-v1-4-convnext-tagger-v2",
                "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                "SmilingWolf/wd-v1-4-vit-tagger-v2",
                "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                "SmilingWolf/wd-v1-4-moat-tagger-v2",
                "SmilingWolf/wd-swinv2-tagger-v3",
                "SmilingWolf/wd-vit-tagger-v3",
                "SmilingWolf/wd-convnext-tagger-v3",
                "SmilingWolf/wd-eva02-large-tagger-v3",
            ]
        );
        assert_eq!(
            Wd14Model::ALL
                .iter()
                .map(|model| model.cache_key())
                .collect::<HashSet<_>>()
                .len(),
            Wd14Model::ALL.len()
        );
        assert!(Wd14Model::ALL.iter().all(|model| !model.label().is_empty()));
    }

    #[test]
    fn selected_model_uses_an_isolated_cache_directory() {
        let root = Path::new("cache");
        let vit = model_files_for(root, Wd14Model::VitV3);
        let eva = model_files_for(root, Wd14Model::Eva02LargeV3);
        assert_eq!(vit.model, root.join("wd-vit-tagger-v3").join("model.onnx"));
        assert_eq!(
            vit.labels,
            root.join("wd-vit-tagger-v3").join("selected_tags.csv")
        );
        assert_ne!(vit.model, eva.model);
    }

    #[test]
    fn model_selection_round_trips_through_preferences_json() {
        for model in Wd14Model::ALL {
            let json = serde_json::to_string(&model).unwrap();
            let restored: Wd14Model = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, model);
        }
        assert_eq!(
            serde_json::to_string(&Wd14Model::Eva02LargeV3).unwrap(),
            "\"wd-eva02-large-tagger-v3\""
        );
        assert_eq!(
            Wd14Model::from_id("SmilingWolf/wd-vit-tagger-v3"),
            Some(Wd14Model::VitV3)
        );
    }

    #[test]
    fn preprocessing_centres_on_white_and_outputs_bgr_hwc() {
        let mut source = RgbaImage::new(1, 2);
        source.put_pixel(0, 0, Rgba([10, 20, 30, 255]));
        source.put_pixel(0, 1, Rgba([100, 50, 0, 128]));
        let pixels = preprocess_image(&DynamicImage::ImageRgba8(source), 2).unwrap();
        assert_eq!(pixels.len(), 12);
        // Like np.pad in kohya_ss, an odd padding pixel is placed right/bottom.
        assert_eq!(&pixels[0..3], &[30.0, 20.0, 10.0]);
        assert_eq!(&pixels[3..6], &[255.0, 255.0, 255.0]);
        assert_eq!(&pixels[6..9], &[127.0, 152.0, 177.0]);
        assert_eq!(&pixels[9..12], &[255.0, 255.0, 255.0]);
    }

    #[test]
    fn parses_all_wd14_categories_in_original_order() {
        let csv = b"tag_id,name,category,count\n1,safe,9,1\n2,1girl,0,1\n3,alice,4,1\n";
        let labels = parse_labels(Cursor::new(csv)).unwrap();
        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0].category, TagCategory::Rating);
        assert_eq!(labels[1].name, "1girl");
        assert_eq!(labels[2].category, TagCategory::Character);
    }

    #[test]
    fn rejects_malformed_or_unknown_label_csv() {
        assert!(parse_labels(Cursor::new(b"name,category\na,0\n")).is_err());
        assert!(parse_labels(Cursor::new(b"tag_id,name,category,count\n1,a,7,1\n")).is_err());
    }

    #[test]
    fn thresholds_rating_and_confidence_sort_are_independent() {
        let labels = vec![
            Label {
                name: "safe".into(),
                category: TagCategory::Rating,
            },
            Label {
                name: "explicit".into(),
                category: TagCategory::Rating,
            },
            Label {
                name: "solo".into(),
                category: TagCategory::General,
            },
            Label {
                name: "alice".into(),
                category: TagCategory::Character,
            },
        ];
        let options = TaggerOptions {
            general_threshold: 0.35,
            character_threshold: 0.85,
            include_rating: true,
        };
        let result = select_predictions(&labels, &[0.1, 0.8, 0.9, 0.84], options).unwrap();
        assert_eq!(
            result.iter().map(|p| p.tag.as_str()).collect::<Vec<_>>(),
            vec!["solo", "explicit"]
        );
        assert!(select_predictions(&labels, &[0.1], options).is_err());
    }

    #[test]
    fn ratings_can_be_omitted() {
        let labels = vec![
            Label {
                name: "safe".into(),
                category: TagCategory::Rating,
            },
            Label {
                name: "tag".into(),
                category: TagCategory::General,
            },
        ];
        let result = select_predictions(
            &labels,
            &[0.99, 0.5],
            TaggerOptions {
                include_rating: false,
                ..TaggerOptions::default()
            },
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tag, "tag");
    }

    /// Opt-in smoke test for release verification; ordinary test runs never
    /// require the 388 MB model or network access.
    #[test]
    #[ignore = "requires TAGGER_NEO_REAL_MODEL_DIR and TAGGER_NEO_REAL_IMAGE"]
    fn real_model_loads_and_predicts_one_image() {
        let model_dir = std::env::var_os("TAGGER_NEO_REAL_MODEL_DIR")
            .expect("TAGGER_NEO_REAL_MODEL_DIR is required");
        let image =
            std::env::var_os("TAGGER_NEO_REAL_IMAGE").expect("TAGGER_NEO_REAL_IMAGE is required");
        let tagger = Wd14Tagger::load(model_dir).unwrap();
        let predictions = tagger
            .predict_path(
                image,
                TaggerOptions {
                    general_threshold: 0.35,
                    character_threshold: 0.35,
                    include_rating: true,
                },
            )
            .unwrap();
        assert!(!predictions.is_empty());
        assert!(predictions
            .windows(2)
            .all(|pair| { pair[0].confidence >= pair[1].confidence }));
        eprintln!(
            "top WD14 predictions: {:?}",
            &predictions[..predictions.len().min(8)]
        );
    }

    /// Opt-in test for the model-specific cache/download path.
    #[test]
    #[ignore = "requires TAGGER_NEO_REAL_MODEL_ROOT, TAGGER_NEO_REAL_MODEL and TAGGER_NEO_REAL_IMAGE"]
    fn selected_real_model_loads_and_predicts_one_image() {
        let model_root = std::env::var_os("TAGGER_NEO_REAL_MODEL_ROOT")
            .expect("TAGGER_NEO_REAL_MODEL_ROOT is required");
        let model_id =
            std::env::var("TAGGER_NEO_REAL_MODEL").expect("TAGGER_NEO_REAL_MODEL is required");
        let model = Wd14Model::from_id(&model_id).expect("unknown TAGGER_NEO_REAL_MODEL");
        let image =
            std::env::var_os("TAGGER_NEO_REAL_IMAGE").expect("TAGGER_NEO_REAL_IMAGE is required");
        let tagger = Wd14Tagger::load_model(model_root, model).unwrap();
        let predictions = tagger
            .predict_path(image, TaggerOptions::default())
            .unwrap();
        assert!(!predictions.is_empty());
    }
}
