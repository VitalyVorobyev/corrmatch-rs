use clap::Parser;
use corrmatch::io::load_gray_image;
use corrmatch::{
    CompileConfig, CompileConfigNoRot, CompiledTemplate, Match, MatchConfig, Matcher, Metric,
    RotationMode, Template,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

const SCHEMA_JSON: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/config.schema.json"));
const EXAMPLE_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/config.example.json"));

#[derive(Parser, Debug)]
#[command(author, version, about = "CorrMatch CLI (JSON config driven)")]
struct Cli {
    /// Path to the JSON configuration file.
    #[arg(short, long, value_name = "FILE", default_value = "config.json")]
    config: PathBuf,
    /// Print the JSON schema and exit.
    #[arg(long)]
    print_schema: bool,
    /// Print an example config and exit.
    #[arg(long)]
    print_example: bool,
    /// Enable tracing output for performance profiling.
    #[arg(long)]
    trace: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum MetricConfig {
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RotationModeConfig {
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

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
#[serde(default)]
struct Config {
    image_path: String,
    template_path: String,
    output_path: Option<String>,
    topk: usize,
    compile: CompileConfigJson,
    #[serde(rename = "match")]
    match_cfg: MatchConfigJson,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            image_path: String::new(),
            template_path: String::new(),
            output_path: None,
            topk: 1,
            compile: CompileConfigJson::default(),
            match_cfg: MatchConfigJson::default(),
        }
    }
}

#[derive(Debug, Serialize)]
struct MatchRecord {
    x: f32,
    y: f32,
    angle_deg: f32,
    score: f32,
}

impl From<Match> for MatchRecord {
    fn from(value: Match) -> Self {
        Self {
            x: value.x,
            y: value.y,
            angle_deg: value.angle_deg,
            score: value.score,
        }
    }
}

#[derive(Debug, Serialize)]
struct Output {
    best: Option<MatchRecord>,
    topk: Vec<MatchRecord>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    if cli.trace {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env().add_directive("corrmatch=info".parse()?))
            .with_target(false)
            .init();
    }

    if cli.print_schema {
        println!("{SCHEMA_JSON}");
        return Ok(());
    }
    if cli.print_example {
        println!("{EXAMPLE_JSON}");
        return Ok(());
    }

    let config_text = fs::read_to_string(&cli.config)?;
    let config: Config = serde_json::from_str(&config_text)?;
    if config.image_path.is_empty() || config.template_path.is_empty() {
        return Err("image_path and template_path must be set in the config".into());
    }
    if config.topk == 0 {
        return Err("topk must be at least 1".into());
    }

    let image = load_gray_image(&config.image_path)?;
    let template_img = load_gray_image(&config.template_path)?;
    let template = Template::new(
        template_img.data().to_vec(),
        template_img.width(),
        template_img.height(),
    )?;

    let compiled = match config.match_cfg.rotation {
        RotationModeConfig::Enabled => CompiledTemplate::compile_rotated(
            &template,
            CompileConfig {
                max_levels: config.compile.max_levels,
                coarse_step_deg: config.compile.coarse_step_deg,
                min_step_deg: config.compile.min_step_deg,
                fill_value: config.compile.fill_value,
                precompute_coarsest: config.compile.precompute_coarsest,
            },
        )?,
        RotationModeConfig::Disabled => CompiledTemplate::compile_unrotated(
            &template,
            CompileConfigNoRot {
                max_levels: config.compile.max_levels,
            },
        )?,
    };

    let matcher = Matcher::new(compiled).with_config(MatchConfig {
        metric: config.match_cfg.metric.into(),
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

    let image_view = image.view();
    let matches = matcher.match_image_topk(image_view, config.topk)?;
    let best = matches.first().copied().map(MatchRecord::from);
    let topk = matches.into_iter().map(MatchRecord::from).collect();
    let output = Output { best, topk };
    let json = serde_json::to_string_pretty(&output)?;

    match config.output_path {
        Some(path) => fs::write(path, json)?,
        None => println!("{json}"),
    }

    Ok(())
}
