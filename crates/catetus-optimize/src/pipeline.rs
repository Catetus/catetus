//! Pipeline runner: executes an ordered list of passes and collects per-pass
//! statistics into a `PipelineReport`.

use anyhow::Result;
use catetus_core::SplatScene;
use serde::Serialize;

use crate::passes::{Pass, PassContext, PassStats};

/// Sequential pipeline of optimization passes.
pub struct Pipeline {
    /// The ordered set of passes that will run.
    pub passes: Vec<Box<dyn Pass>>,
}

/// Aggregate report for a full pipeline run.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineReport {
    /// Splat count before the pipeline ran.
    pub splats_before: usize,
    /// Splat count after the pipeline ran.
    pub splats_after: usize,
    /// Per-pass stats in execution order.
    pub passes: Vec<NamedPassStats>,
}

/// A `PassStats` value tagged with the pass name that produced it.
#[derive(Debug, Clone, Serialize)]
pub struct NamedPassStats {
    /// Pass identifier.
    pub name: String,
    /// Stats produced by that pass.
    pub stats: PassStats,
}

impl Pipeline {
    /// Construct a new empty pipeline.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Append a pass.
    pub fn push(mut self, p: Box<dyn Pass>) -> Self {
        self.passes.push(p);
        self
    }

    /// Run every pass in order, capturing stats.
    pub fn run(&self, scene: &mut SplatScene) -> Result<PipelineReport> {
        self.run_with_progress(scene, |_, _, _| {})
    }

    /// Like `run` but invokes `progress(pass_index, total_passes, pass_name)`
    /// before each pass starts and once more after the last pass with
    /// `(total_passes, total_passes, "done")`. Lets callers stream
    /// pass-level progress to a UI / subprocess pipe without coupling the
    /// pipeline to any particular sink.
    pub fn run_with_progress<F>(
        &self,
        scene: &mut SplatScene,
        progress: F,
    ) -> Result<PipelineReport>
    where
        F: FnMut(usize, usize, &str),
    {
        self.run_with_progress_and_ctx(scene, PassContext::default(), progress)
    }

    /// Like `run_with_progress` but accepts a caller-supplied initial
    /// `PassContext`. Used by the CLI to inject the optional per-splat
    /// SH-rest Jacobian weights (from `--jacobian-sidecar`) into
    /// `VQPaletteShRest`'s weighted Lloyd's update path. The context's
    /// `sh_rest_weights` (if `Some`) MUST satisfy
    /// `weights.len() == scene.splats.len()` at call time — structural
    /// passes (`RemoveInvalidSplats`, `MortonSort`) keep it in lockstep
    /// from there.
    pub fn run_with_progress_and_ctx<F>(
        &self,
        scene: &mut SplatScene,
        ctx: PassContext,
        progress: F,
    ) -> Result<PipelineReport>
    where
        F: FnMut(usize, usize, &str),
    {
        let (report, _ctx) = self.run_with_progress_returning_ctx(scene, ctx, progress)?;
        Ok(report)
    }

    /// Variant of [`run_with_progress_and_ctx`] that returns the post-run
    /// `PassContext` alongside the report. Used by the V5.2 sidecar
    /// encoder in the CLI to drain the joint-Jacobian array AFTER the
    /// structural passes (`RemoveInvalidSplats`, `MortonSort`) have kept
    /// it in lockstep with the now-filtered/reordered splat array.
    pub fn run_with_progress_returning_ctx<F>(
        &self,
        scene: &mut SplatScene,
        ctx: PassContext,
        mut progress: F,
    ) -> Result<(PipelineReport, PassContext)>
    where
        F: FnMut(usize, usize, &str),
    {
        let mut ctx = ctx;
        let before = scene.splats.len();
        let total = self.passes.len();
        let mut passes = Vec::with_capacity(total);
        for (i, p) in self.passes.iter().enumerate() {
            progress(i, total, p.name());
            let stats = p.run(scene, &mut ctx)?;
            passes.push(NamedPassStats {
                name: p.name().to_string(),
                stats,
            });
        }
        progress(total, total, "done");
        let after = scene.splats.len();
        Ok((
            PipelineReport {
                splats_before: before,
                splats_after: after,
                passes,
            },
            ctx,
        ))
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
