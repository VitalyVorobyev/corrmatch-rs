//! Integration tests validating the matcher against synthetic test cases.
//!
//! These tests load ground-truth synthetic cases and verify that the matcher
//! produces results within acceptable tolerances.

use corrmatch::{
    CompileConfig, CompileConfigNoRot, CompiledTemplate, MatchConfig, Matcher, Metric,
    RotationMode, Template,
};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// Position tolerance in pixels.
const POSITION_TOLERANCE_PX: f32 = 3.0;

/// Angle tolerance in degrees.
const ANGLE_TOLERANCE_DEG: f32 = 2.0;

/// Minimum ZNCC score threshold for valid matches.
const MIN_SCORE_THRESHOLD: f32 = 0.8;
const MIN_SCORE_THRESHOLD_OCCLUDED_25PCT: f32 = 0.77;

fn min_score_threshold(case_id: &str) -> f32 {
    match case_id {
        "occluded_25pct" => MIN_SCORE_THRESHOLD_OCCLUDED_25PCT,
        _ => MIN_SCORE_THRESHOLD,
    }
}

/// Manifest entry for a synthetic test case.
#[derive(Debug, Deserialize)]
struct ManifestEntry {
    case_id: String,
    dir: String,
    #[allow(dead_code)]
    present: bool,
}

/// Manifest file structure.
#[derive(Debug, Deserialize)]
struct Manifest {
    cases: Vec<ManifestEntry>,
}

/// Ground truth instance in meta.json.
#[derive(Debug, Deserialize)]
struct Instance {
    x: f32,
    y: f32,
    #[serde(default)]
    angle_deg: f32,
}

/// Meta.json structure (partial - only what we need).
#[derive(Debug, Deserialize)]
struct Meta {
    case_id: String,
    #[serde(default)]
    present: bool,
    #[serde(default)]
    instances: Vec<Instance>,
}

/// CLI config structure (matches cli_config.json format).
#[derive(Debug, Deserialize)]
#[serde(default)]
struct CompileConfigJson {
    max_levels: usize,
    coarse_step_deg: f32,
    min_step_deg: f32,
    fill_value: u8,
    precompute_coarsest: bool,
}

impl Default for CompileConfigJson {
    fn default() -> Self {
        let cfg = CompileConfig::default();
        Self {
            max_levels: cfg.max_levels,
            coarse_step_deg: cfg.coarse_step_deg,
            min_step_deg: cfg.min_step_deg,
            fill_value: cfg.fill_value,
            precompute_coarsest: cfg.precompute_coarsest,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
enum MetricConfig {
    #[default]
    Zncc,
    Ssd,
}

impl From<MetricConfig> for Metric {
    fn from(value: MetricConfig) -> Self {
        match value {
            MetricConfig::Zncc => Metric::Zncc,
            MetricConfig::Ssd => Metric::Ssd,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
#[serde(rename_all = "snake_case")]
enum RotationModeConfig {
    #[default]
    Disabled,
    Enabled,
}

impl From<RotationModeConfig> for RotationMode {
    fn from(value: RotationModeConfig) -> Self {
        match value {
            RotationModeConfig::Disabled => RotationMode::Disabled,
            RotationModeConfig::Enabled => RotationMode::Enabled,
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
struct MatchConfigJson {
    metric: MetricConfig,
    rotation: RotationModeConfig,
    parallel: bool,
    max_image_levels: usize,
    beam_width: usize,
    per_angle_topk: usize,
    nms_radius: usize,
    roi_radius: usize,
    angle_half_range_steps: usize,
    min_var_i: f32,
    min_score: f32,
}

impl Default for MatchConfigJson {
    fn default() -> Self {
        let cfg = MatchConfig::default();
        Self {
            metric: MetricConfig::Zncc,
            rotation: RotationModeConfig::Disabled,
            parallel: cfg.parallel,
            max_image_levels: cfg.max_image_levels,
            beam_width: cfg.beam_width,
            per_angle_topk: cfg.per_angle_topk,
            nms_radius: cfg.nms_radius,
            roi_radius: cfg.roi_radius,
            angle_half_range_steps: cfg.angle_half_range_steps,
            min_var_i: cfg.min_var_i,
            min_score: cfg.min_score,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default)]
struct CliConfig {
    compile: CompileConfigJson,
    #[serde(rename = "match")]
    match_cfg: MatchConfigJson,
}

/// Loads a grayscale PNG image and returns (data, width, height).
#[cfg(feature = "image-io")]
fn load_image(path: &Path) -> (Vec<u8>, usize, usize) {
    use corrmatch::io::load_gray_image;
    let img = load_gray_image(path.to_str().unwrap()).expect("Failed to load image");
    (img.data().to_vec(), img.width(), img.height())
}

#[cfg(not(feature = "image-io"))]
fn load_image(_path: &Path) -> (Vec<u8>, usize, usize) {
    panic!("image-io feature required for synthetic tests");
}

/// Returns the path to the synthetic_cases directory.
fn synthetic_cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("synthetic_cases")
}

/// Discovers all synthetic test cases from manifest.
fn discover_cases() -> Vec<(String, PathBuf)> {
    let dir = synthetic_cases_dir();
    let manifest_path = dir.join("manifest.json");

    if !manifest_path.exists() {
        eprintln!("Warning: manifest.json not found at {:?}", manifest_path);
        return vec![];
    }

    let manifest_text = fs::read_to_string(&manifest_path).expect("Failed to read manifest");
    let manifest: Manifest =
        serde_json::from_str(&manifest_text).expect("Failed to parse manifest");

    manifest
        .cases
        .into_iter()
        .map(|entry| (entry.case_id, dir.join(entry.dir)))
        .collect()
}

/// Wraps angle difference to [-180, 180].
fn wrap_angle_diff(a: f32, b: f32) -> f32 {
    let mut diff = (a - b) % 360.0;
    if diff > 180.0 {
        diff -= 360.0;
    } else if diff < -180.0 {
        diff += 360.0;
    }
    diff.abs()
}

/// Runs a single synthetic test case.
fn run_case(case_dir: &Path) -> Result<(), String> {
    // Load meta.json
    let meta_path = case_dir.join("meta.json");
    let meta_text =
        fs::read_to_string(&meta_path).map_err(|e| format!("Failed to read meta.json: {}", e))?;
    let meta: Meta = serde_json::from_str(&meta_text)
        .map_err(|e| format!("Failed to parse meta.json: {}", e))?;

    let score_threshold = min_score_threshold(&meta.case_id);

    // Load cli_config.json
    let config_path = case_dir.join("cli_config.json");
    let config_text = fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read cli_config.json: {}", e))?;
    let config: CliConfig = serde_json::from_str(&config_text)
        .map_err(|e| format!("Failed to parse cli_config.json: {}", e))?;

    // Load images
    let image_path = case_dir.join("image.png");
    let template_path = case_dir.join("template.png");

    let (img_data, img_w, img_h) = load_image(&image_path);
    let (tpl_data, tpl_w, tpl_h) = load_image(&template_path);

    // Create template
    let template = Template::new(tpl_data, tpl_w, tpl_h)
        .map_err(|e| format!("Failed to create template: {}", e))?;

    // Compile template based on rotation mode
    let rotation_enabled = matches!(config.match_cfg.rotation, RotationModeConfig::Enabled);

    let compiled = if rotation_enabled {
        CompiledTemplate::compile_rotated(
            &template,
            CompileConfig {
                max_levels: config.compile.max_levels,
                coarse_step_deg: config.compile.coarse_step_deg,
                min_step_deg: config.compile.min_step_deg,
                fill_value: config.compile.fill_value,
                precompute_coarsest: config.compile.precompute_coarsest,
            },
        )
        .map_err(|e| format!("Failed to compile template: {}", e))?
    } else {
        CompiledTemplate::compile_unrotated(
            &template,
            CompileConfigNoRot {
                max_levels: config.compile.max_levels,
            },
        )
        .map_err(|e| format!("Failed to compile template: {}", e))?
    };

    // Create matcher with config
    let matcher = Matcher::new(compiled).with_config(MatchConfig {
        metric: config.match_cfg.metric.clone().into(),
        rotation: config.match_cfg.rotation.into(),
        parallel: config.match_cfg.parallel,
        max_image_levels: config.match_cfg.max_image_levels,
        beam_width: config.match_cfg.beam_width,
        per_angle_topk: config.match_cfg.per_angle_topk,
        nms_radius: config.match_cfg.nms_radius,
        roi_radius: config.match_cfg.roi_radius,
        angle_half_range_steps: config.match_cfg.angle_half_range_steps,
        min_var_i: config.match_cfg.min_var_i,
        min_score: config.match_cfg.min_score,
    });

    // Create image view
    let image_view = corrmatch::ImageView::from_slice(&img_data, img_w, img_h)
        .map_err(|e| format!("Failed to create image view: {}", e))?;

    // Run matching
    let matches = matcher
        .match_image_topk(image_view, 1)
        .map_err(|e| format!("Matching failed: {}", e))?;

    // Validate results
    if !meta.present {
        // Negative case: score should be low
        if let Some(result) = matches.first() {
            if matches!(config.match_cfg.metric, MetricConfig::Zncc)
                && result.score >= score_threshold
            {
                return Err(format!(
                    "Negative case has unexpectedly high score: {:.4}",
                    result.score
                ));
            }
        }
        return Ok(());
    }

    // Positive case: validate against ground truth
    if meta.instances.is_empty() {
        return Err("No ground truth instances in meta.json".to_string());
    }

    let result = matches
        .first()
        .ok_or_else(|| "No match found".to_string())?;

    let expected = &meta.instances[0];

    // Check position
    let dx = (result.x - expected.x).abs();
    let dy = (result.y - expected.y).abs();

    if dx > POSITION_TOLERANCE_PX {
        return Err(format!(
            "x error: {:.2} > {:.1} (got {:.2}, expected {:.2})",
            dx, POSITION_TOLERANCE_PX, result.x, expected.x
        ));
    }

    if dy > POSITION_TOLERANCE_PX {
        return Err(format!(
            "y error: {:.2} > {:.1} (got {:.2}, expected {:.2})",
            dy, POSITION_TOLERANCE_PX, result.y, expected.y
        ));
    }

    // Check angle (only if rotation enabled)
    if rotation_enabled {
        let da = wrap_angle_diff(result.angle_deg, expected.angle_deg);
        if da > ANGLE_TOLERANCE_DEG {
            return Err(format!(
                "angle error: {:.2} > {:.1} (got {:.2}, expected {:.2})",
                da, ANGLE_TOLERANCE_DEG, result.angle_deg, expected.angle_deg
            ));
        }
    }

    // Check score for ZNCC
    if matches!(config.match_cfg.metric, MetricConfig::Zncc) && result.score < score_threshold {
        return Err(format!(
            "Score {:.4} below threshold {:.4}",
            result.score, score_threshold
        ));
    }

    Ok(())
}

#[test]
#[cfg(feature = "image-io")]
fn test_synthetic_cases() {
    let cases = discover_cases();

    if cases.is_empty() {
        eprintln!("No synthetic cases found. Generate them with:");
        eprintln!(
            "  python tools/synth_cases/generate_cases.py --suite standard --out ./synthetic_cases"
        );
        return;
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut failures: Vec<(String, String)> = vec![];

    for (case_id, case_dir) in &cases {
        match run_case(case_dir) {
            Ok(()) => {
                passed += 1;
                println!("PASS: {}", case_id);
            }
            Err(e) => {
                failed += 1;
                println!("FAIL: {} - {}", case_id, e);
                failures.push((case_id.clone(), e));
            }
        }
    }

    println!("\n--- Summary ---");
    println!("Passed: {}/{}", passed, cases.len());
    println!("Failed: {}/{}", failed, cases.len());

    if !failures.is_empty() {
        println!("\nFailures:");
        for (case_id, error) in &failures {
            println!("  {}: {}", case_id, error);
        }
        panic!("{} test case(s) failed", failures.len());
    }
}
