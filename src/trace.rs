//! Conditional tracing macros (zero-cost when feature disabled).
//!
//! This module provides macros that emit tracing spans and events when the
//! `tracing` feature is enabled, and compile to nothing when disabled.

/// Create an info-level span for a major operation.
///
/// When the `tracing` feature is enabled, this creates a `tracing::info_span!`.
/// When disabled, it compiles to a no-op that returns a dummy guard.
#[cfg(feature = "tracing")]
macro_rules! trace_span {
    ($name:expr $(, $($field:tt)*)?) => {
        tracing::info_span!($name $(, $($field)*)?)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_span {
    ($name:expr $(, $($field:tt)*)?) => {
        $crate::trace::NoopSpan
    };
}

/// Emit an info-level event for key measurements.
///
/// When the `tracing` feature is enabled, this calls `tracing::info!`.
/// When disabled, it compiles to nothing (values are evaluated but discarded to avoid
/// unused warnings).
#[cfg(feature = "tracing")]
macro_rules! trace_event {
    ($name:expr, $($key:ident = $value:expr),+ $(,)?) => {
        tracing::info!(name: $name, $($key = $value),+)
    };
    ($name:expr) => {
        tracing::info!(name: $name)
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace_event {
    ($name:expr, $($key:ident = $value:expr),+ $(,)?) => {
        // Evaluate expressions to silence unused warnings, but discard results
        let _ = ($($value,)+);
    };
    ($name:expr) => {};
}

pub(crate) use trace_event;
pub(crate) use trace_span;

/// A no-op span guard used when tracing is disabled.
///
/// This struct exists so that `trace_span!` can be used in `let _guard = trace_span!(...).entered();`
/// patterns without conditional compilation at call sites.
#[cfg(not(feature = "tracing"))]
pub struct NoopSpan;

#[cfg(not(feature = "tracing"))]
impl NoopSpan {
    /// Returns self, mimicking `Span::entered()`.
    #[inline]
    pub fn entered(self) -> Self {
        self
    }
}
