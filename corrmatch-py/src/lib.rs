//! Python bindings for corrmatch template matching library.
//!
//! This module exposes the high-level corrmatch API to Python via PyO3.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use corrmatch::{
    CompileConfig as RustCompileConfig, CompileConfigNoRot as RustCompileConfigNoRot,
    CompiledTemplate as RustCompiledTemplate, CorrMatchError, ImageView, Match as RustMatch,
    MatchConfig as RustMatchConfig, Matcher as RustMatcher, Metric as RustMetric,
    RotationMode as RustRotationMode, Template as RustTemplate,
};

/// Convert a CorrMatchError to a Python exception.
fn to_py_err(err: CorrMatchError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Match result containing position, angle, and score.
#[pyclass]
#[derive(Clone)]
pub struct Match {
    /// Top-left x coordinate of the template placement.
    #[pyo3(get)]
    pub x: f32,
    /// Top-left y coordinate of the template placement.
    #[pyo3(get)]
    pub y: f32,
    /// Rotation angle in degrees.
    #[pyo3(get)]
    pub angle_deg: f32,
    /// Match score (ZNCC in [-1, 1] or negative SSD).
    #[pyo3(get)]
    pub score: f32,
}

#[pymethods]
impl Match {
    fn __repr__(&self) -> String {
        format!(
            "Match(x={:.2}, y={:.2}, angle_deg={:.2}, score={:.4})",
            self.x, self.y, self.angle_deg, self.score
        )
    }
}

impl From<RustMatch> for Match {
    fn from(m: RustMatch) -> Self {
        Self {
            x: m.x,
            y: m.y,
            angle_deg: m.angle_deg,
            score: m.score,
        }
    }
}

/// Configuration for compiling a template with rotation support.
#[pyclass]
#[derive(Clone)]
pub struct CompileConfig {
    inner: RustCompileConfig,
}

#[pymethods]
impl CompileConfig {
    /// Create a new CompileConfig.
    ///
    /// Args:
    ///     max_levels: Maximum pyramid levels (default: 6)
    ///     coarse_step_deg: Coarse rotation step in degrees (default: 10.0)
    ///     min_step_deg: Minimum rotation step in degrees (default: 0.5)
    ///     fill_value: Fill value for out-of-bounds rotations (default: 0)
    ///     precompute_coarsest: Precompute coarsest level rotations (default: True)
    #[new]
    #[pyo3(signature = (max_levels=6, coarse_step_deg=10.0, min_step_deg=0.5, fill_value=0, precompute_coarsest=true))]
    fn new(
        max_levels: usize,
        coarse_step_deg: f32,
        min_step_deg: f32,
        fill_value: u8,
        precompute_coarsest: bool,
    ) -> PyResult<Self> {
        let inner = RustCompileConfig {
            max_levels,
            coarse_step_deg,
            min_step_deg,
            fill_value,
            precompute_coarsest,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Validate the configuration.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "CompileConfig(max_levels={}, coarse_step_deg={}, min_step_deg={}, fill_value={}, precompute_coarsest={})",
            self.inner.max_levels,
            self.inner.coarse_step_deg,
            self.inner.min_step_deg,
            self.inner.fill_value,
            self.inner.precompute_coarsest
        )
    }
}

/// Configuration for the matching process.
#[pyclass]
#[derive(Clone)]
pub struct MatchConfig {
    inner: RustMatchConfig,
}

#[pymethods]
impl MatchConfig {
    /// Create a new MatchConfig.
    ///
    /// Args:
    ///     metric: "zncc" or "ssd" (default: "zncc")
    ///     rotation: "enabled" or "disabled" (default: "disabled")
    ///     parallel: Enable parallel execution (default: False)
    ///     max_image_levels: Maximum image pyramid levels (default: 6)
    ///     beam_width: Candidates kept per level (default: 8)
    ///     per_angle_topk: Top peaks per angle at coarsest level (default: 3)
    ///     nms_radius: Non-maximum suppression radius (default: 6)
    ///     roi_radius: Refinement ROI radius (default: 8)
    ///     angle_half_range_steps: Angle search half-range in steps (default: 1)
    ///     min_var_i: Minimum image patch variance for ZNCC (default: 1e-8)
    ///     min_score: Minimum score threshold (default: -inf)
    #[new]
    #[pyo3(signature = (
        metric = "zncc",
        rotation = "disabled",
        parallel = false,
        max_image_levels = 6,
        beam_width = 8,
        per_angle_topk = 3,
        nms_radius = 6,
        roi_radius = 8,
        angle_half_range_steps = 1,
        min_var_i = 1e-8,
        min_score = f32::NEG_INFINITY
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        metric: &str,
        rotation: &str,
        parallel: bool,
        max_image_levels: usize,
        beam_width: usize,
        per_angle_topk: usize,
        nms_radius: usize,
        roi_radius: usize,
        angle_half_range_steps: usize,
        min_var_i: f32,
        min_score: f32,
    ) -> PyResult<Self> {
        let metric = match metric.to_lowercase().as_str() {
            "zncc" => RustMetric::Zncc,
            "ssd" => RustMetric::Ssd,
            _ => return Err(PyValueError::new_err("metric must be 'zncc' or 'ssd'")),
        };
        let rotation = match rotation.to_lowercase().as_str() {
            "enabled" => RustRotationMode::Enabled,
            "disabled" => RustRotationMode::Disabled,
            _ => {
                return Err(PyValueError::new_err(
                    "rotation must be 'enabled' or 'disabled'",
                ))
            }
        };
        let inner = RustMatchConfig {
            metric,
            rotation,
            parallel,
            max_image_levels,
            beam_width,
            per_angle_topk,
            nms_radius,
            roi_radius,
            angle_half_range_steps,
            min_var_i,
            min_score,
        };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Validate the configuration.
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        let metric = match self.inner.metric {
            RustMetric::Zncc => "zncc",
            RustMetric::Ssd => "ssd",
        };
        let rotation = match self.inner.rotation {
            RustRotationMode::Enabled => "enabled",
            RustRotationMode::Disabled => "disabled",
        };
        format!(
            "MatchConfig(metric='{}', rotation='{}', parallel={}, beam_width={})",
            metric, rotation, self.inner.parallel, self.inner.beam_width
        )
    }
}

/// A grayscale template for matching.
#[pyclass]
pub struct Template {
    inner: RustTemplate,
}

#[pymethods]
impl Template {
    /// Create a template from a 2D numpy array.
    ///
    /// Args:
    ///     pixels: 2D uint8 numpy array (height x width)
    #[new]
    fn new(pixels: PyReadonlyArray2<'_, u8>) -> PyResult<Self> {
        let shape = pixels.shape();
        let height = shape[0];
        let width = shape[1];

        // Copy data into contiguous Vec
        let data: Vec<u8> = pixels.as_slice()?.to_vec();

        let inner = RustTemplate::new(data, width, height).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Load a template from an image file.
    ///
    /// Args:
    ///     path: Path to grayscale or RGB image file
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let owned = corrmatch::io::load_gray_image(path).map_err(to_py_err)?;
        let inner = RustTemplate::new(owned.data().to_vec(), owned.width(), owned.height())
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Compile the template with rotation support.
    ///
    /// Args:
    ///     config: CompileConfig (default: CompileConfig())
    #[pyo3(signature = (config = None))]
    fn compile(&self, config: Option<CompileConfig>) -> PyResult<CompiledTemplate> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        cfg.validate().map_err(to_py_err)?;
        let compiled =
            RustCompiledTemplate::compile_rotated(&self.inner, cfg).map_err(to_py_err)?;
        let num_levels = compiled.num_levels();
        let matcher = RustMatcher::new(compiled);
        Ok(CompiledTemplate::new_with_matcher(matcher, num_levels))
    }

    /// Compile the template without rotation support (faster).
    ///
    /// Args:
    ///     max_levels: Maximum pyramid levels (default: 6)
    #[pyo3(signature = (max_levels = 6))]
    fn compile_no_rotation(&self, max_levels: usize) -> PyResult<CompiledTemplate> {
        let cfg = RustCompileConfigNoRot { max_levels };
        let compiled =
            RustCompiledTemplate::compile_unrotated(&self.inner, cfg).map_err(to_py_err)?;
        let num_levels = compiled.num_levels();
        let matcher = RustMatcher::new(compiled);
        Ok(CompiledTemplate::new_with_matcher(matcher, num_levels))
    }

    /// Get template width.
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }

    /// Get template height.
    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }

    fn __repr__(&self) -> String {
        format!("Template({}x{})", self.inner.width(), self.inner.height())
    }
}

/// A compiled template ready for matching.
///
/// This class wraps a Rust Matcher directly since CompiledTemplate
/// cannot be shared across Python boundaries (no Clone).
#[pyclass]
pub struct CompiledTemplate {
    // Store the matcher directly since CompiledTemplate is consumed
    matcher: Option<RustMatcher>,
    num_levels: usize,
}

impl CompiledTemplate {
    fn new_with_matcher(matcher: RustMatcher, num_levels: usize) -> Self {
        Self {
            matcher: Some(matcher),
            num_levels,
        }
    }
}

#[pymethods]
impl CompiledTemplate {
    /// Get the number of pyramid levels.
    #[getter]
    fn num_levels(&self) -> usize {
        self.num_levels
    }

    /// Create a matcher from this compiled template.
    ///
    /// Note: This consumes the compiled template. Create a new one for
    /// different configurations.
    ///
    /// Args:
    ///     config: MatchConfig (default: MatchConfig())
    #[pyo3(signature = (config = None))]
    fn matcher(&mut self, config: Option<MatchConfig>) -> PyResult<Matcher> {
        let inner = self.matcher.take().ok_or_else(|| {
            PyRuntimeError::new_err("CompiledTemplate already consumed - recompile the template")
        })?;

        let cfg = config.map(|c| c.inner).unwrap_or_default();
        cfg.validate().map_err(to_py_err)?;
        let inner = inner.with_config(cfg);

        Ok(Matcher { inner })
    }

    fn __repr__(&self) -> String {
        let status = if self.matcher.is_some() {
            "ready"
        } else {
            "consumed"
        };
        format!(
            "CompiledTemplate(num_levels={}, status='{}')",
            self.num_levels, status
        )
    }
}

/// Template matcher that performs coarse-to-fine search.
#[pyclass]
pub struct Matcher {
    inner: RustMatcher,
}

#[pymethods]
impl Matcher {
    /// Match the template against an image, returning the best match.
    ///
    /// Args:
    ///     image: 2D uint8 numpy array (height x width)
    ///
    /// Returns:
    ///     Match object with x, y, angle_deg, and score
    fn match_image(&self, image: PyReadonlyArray2<'_, u8>) -> PyResult<Match> {
        let shape = image.shape();
        let height = shape[0];
        let width = shape[1];
        let data = image.as_slice()?;

        let view = ImageView::from_slice(data, width, height).map_err(to_py_err)?;
        let result = self.inner.match_image(view).map_err(to_py_err)?;
        Ok(result.into())
    }

    /// Match the template against an image, returning top-k matches.
    ///
    /// Args:
    ///     image: 2D uint8 numpy array (height x width)
    ///     k: Number of matches to return
    ///
    /// Returns:
    ///     List of Match objects, sorted by score (best first)
    fn match_topk(&self, image: PyReadonlyArray2<'_, u8>, k: usize) -> PyResult<Vec<Match>> {
        let shape = image.shape();
        let height = shape[0];
        let width = shape[1];
        let data = image.as_slice()?;

        let view = ImageView::from_slice(data, width, height).map_err(to_py_err)?;
        let results = self.inner.match_image_topk(view, k).map_err(to_py_err)?;
        Ok(results.into_iter().map(Match::from).collect())
    }

    fn __repr__(&self) -> String {
        "Matcher()".to_string()
    }
}

/// Convenience function to match a template against an image.
///
/// This is a simplified API that compiles the template and runs matching
/// in a single call. For repeated matching with the same template,
/// use Template.compile() and Matcher separately for better performance.
///
/// Args:
///     image: 2D uint8 numpy array (height x width)
///     template: 2D uint8 numpy array (height x width)
///     rotation: "enabled" or "disabled" (default: "disabled")
///     metric: "zncc" or "ssd" (default: "zncc")
///     parallel: Enable parallel execution (default: False)
///
/// Returns:
///     Match object with x, y, angle_deg, and score
#[pyfunction]
#[pyo3(signature = (image, template, rotation = "disabled", metric = "zncc", parallel = false))]
fn match_template(
    image: PyReadonlyArray2<'_, u8>,
    template: PyReadonlyArray2<'_, u8>,
    rotation: &str,
    metric: &str,
    parallel: bool,
) -> PyResult<Match> {
    // Create template
    let tpl_shape = template.shape();
    let tpl_height = tpl_shape[0];
    let tpl_width = tpl_shape[1];
    let tpl_data: Vec<u8> = template.as_slice()?.to_vec();
    let tpl = RustTemplate::new(tpl_data, tpl_width, tpl_height).map_err(to_py_err)?;

    // Parse rotation mode
    let rotation_mode = match rotation.to_lowercase().as_str() {
        "enabled" => RustRotationMode::Enabled,
        "disabled" => RustRotationMode::Disabled,
        _ => {
            return Err(PyValueError::new_err(
                "rotation must be 'enabled' or 'disabled'",
            ))
        }
    };

    // Compile template
    let compiled = if rotation_mode == RustRotationMode::Enabled {
        RustCompiledTemplate::compile_rotated(&tpl, RustCompileConfig::default())
    } else {
        RustCompiledTemplate::compile_unrotated(&tpl, RustCompileConfigNoRot::default())
    }
    .map_err(to_py_err)?;

    // Parse metric
    let metric_val = match metric.to_lowercase().as_str() {
        "zncc" => RustMetric::Zncc,
        "ssd" => RustMetric::Ssd,
        _ => return Err(PyValueError::new_err("metric must be 'zncc' or 'ssd'")),
    };

    // Create match config
    let match_config = RustMatchConfig {
        metric: metric_val,
        rotation: rotation_mode,
        parallel,
        ..RustMatchConfig::default()
    };

    // Create matcher and run
    let matcher = RustMatcher::new(compiled).with_config(match_config);

    // Parse image
    let img_shape = image.shape();
    let img_height = img_shape[0];
    let img_width = img_shape[1];
    let img_data = image.as_slice()?;
    let view = ImageView::from_slice(img_data, img_width, img_height).map_err(to_py_err)?;

    let result = matcher.match_image(view).map_err(to_py_err)?;
    Ok(result.into())
}

/// Python module for corrmatch template matching.
#[pymodule]
fn _corrmatch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Match>()?;
    m.add_class::<CompileConfig>()?;
    m.add_class::<MatchConfig>()?;
    m.add_class::<Template>()?;
    m.add_class::<CompiledTemplate>()?;
    m.add_class::<Matcher>()?;
    m.add_function(wrap_pyfunction!(match_template, m)?)?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
