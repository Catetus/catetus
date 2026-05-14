//! Pipeline runner: executes an ordered list of passes and collects per-pass
//! statistics into a `PipelineReport`.

use anyhow::Result;
use serde::Serialize;
use splatforge_core::SplatScene;

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
        let mut ctx = PassContext::default();
        let before = scene.splats.len();
        let mut passes = Vec::with_capacity(self.passes.len());
        for p in &self.passes {
            let stats = p.run(scene, &mut ctx)?;
            passes.push(NamedPassStats {
                name: p.name().to_string(),
                stats,
            });
        }
        let after = scene.splats.len();
        Ok(PipelineReport {
            splats_before: before,
            splats_after: after,
            passes,
        })
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new()
    }
}
