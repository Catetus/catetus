//! Tier routing by spherical-harmonic (SH) degree.
//!
//! Catetus' high-fidelity tiers — T2.1.R (`--auto-jacobian`) and V5.2
//! (`--emit-v5-tail`) — work by re-encoding the view-dependent SH-rest
//! coefficients (`f_rest_*`). On captures with **no** view-dependent color
//! (SH degree 0, e.g. Scaniverse's free tier) there are no SH-rest
//! coefficients to re-encode, so those tiers are no-ops and produce output
//! byte-identical to the SF baseline. This was a real confusion in KORIYAMA-1
//! where SF and T2.1.R came out byte-identical with no explanation.
//!
//! This module inspects a parsed [`SplatScene`] on ingest and decides which
//! product tier it qualifies for, so the CLI/API can route accordingly and
//! surface an honest, user-facing explanation (plus a recapture upsell for the
//! SH=0 cohort) instead of silently emitting identical output.
//!
//! SH degree is derived from the per-channel SH coefficient count carried in
//! [`Color::Sh`] (`coeffs.len() / 3`), matching the PLY decoder's own
//! `f_rest`-column → degree mapping in `catetus-ply`:
//!
//! | coeffs/channel | f_rest cols | SH degree |
//! |----------------|-------------|-----------|
//! | 1              | 0           | 0         |
//! | 4              | 9          | 1         |
//! | 9              | 24          | 2         |
//! | 16             | 45          | 3         |

use crate::ir::{Color, SplatScene};

/// The product tier a scene qualifies for, derived from its SH degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// SH degree >= 2: full quality tiers available.
    ///
    /// `--auto-jacobian` (T2.1.R) and `--emit-v5-tail` (V5.2) engage the
    /// VQ SH-rest palette meaningfully and deliver their headline gains.
    Full,
    /// SH degree 1: partial — a limited SH-rest budget (9 f_rest columns).
    ///
    /// Quality tiers engage but have far less to work with; gains are muted.
    Partial,
    /// SH degree 0: baseline only.
    ///
    /// No view-dependent color. Quality tiers are no-ops; the SF baseline
    /// (lossless container) is the right choice. Recapturing at SH=3 would
    /// unlock the quality tiers.
    Baseline,
}

impl Tier {
    /// Short stable identifier (useful for JSON / machine consumers).
    pub fn id(self) -> &'static str {
        match self {
            Tier::Full => "full",
            Tier::Partial => "partial",
            Tier::Baseline => "baseline",
        }
    }

    /// Human-readable tier name.
    pub fn name(self) -> &'static str {
        match self {
            Tier::Full => "Full quality tiers",
            Tier::Partial => "Partial (limited SH-rest budget)",
            Tier::Baseline => "SF baseline (lossless)",
        }
    }

    /// Whether the high-fidelity quality tiers (T2.1.R / V5.2) engage
    /// meaningfully on this tier.
    pub fn quality_tiers_engage(self) -> bool {
        matches!(self, Tier::Full | Tier::Partial)
    }
}

/// The recapture upsell line shown for SH=0 captures.
pub const RECAPTURE_UPSELL: &str = "This capture has no view-dependent color (SH=0). \
Catetus compresses it losslessly, but recapturing at SH=3 would unlock ~15 dB better \
quality at the same size.";

/// A full tier-routing decision for an ingested scene.
#[derive(Debug, Clone)]
pub struct TierDecision {
    /// SH degree detected (0/1/2/3), or `None` if the scene is empty / the
    /// per-channel coefficient count is non-standard.
    pub sh_degree: Option<u8>,
    /// Number of SH coefficients per color channel (1 for DC-only / SH0).
    pub coeffs_per_channel: usize,
    /// The tier the scene qualifies for.
    pub tier: Tier,
    /// Whether `--auto-jacobian` (T2.1.R) will have a real effect.
    pub auto_jacobian_effective: bool,
    /// Whether `--emit-v5-tail` (V5.2) will have a real effect.
    pub v5_tail_effective: bool,
    /// One-line, user-facing reason for the routing decision.
    pub reason: String,
}

impl TierDecision {
    /// A short string for the SH degree (e.g. "0", "3", or "unknown").
    pub fn degree_str(&self) -> String {
        self.sh_degree
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".into())
    }
}

/// Derive the per-channel SH coefficient count from the scene's first splat.
///
/// All splats in an Inria-style 3DGS scene share one SH degree, so the first
/// `Color::Sh` we find is authoritative. `Color::Rgb` (and an empty scene)
/// means DC-only — one coefficient per channel, i.e. SH degree 0.
fn coeffs_per_channel(scene: &SplatScene) -> usize {
    match scene.splats.first().map(|s| &s.color) {
        Some(Color::Sh { coeffs, .. }) => (coeffs.len() / 3).max(1),
        // `Color::Rgb` and an empty scene are both DC-only (SH degree 0).
        _ => 1,
    }
}

/// Map a per-channel coefficient count to an SH degree (0..3), or `None` for a
/// non-standard count.
fn degree_from_coeffs(n: usize) -> Option<u8> {
    match n {
        1 => Some(0),
        4 => Some(1),
        9 => Some(2),
        16 => Some(3),
        _ => None,
    }
}

/// Inspect a parsed scene and decide which tier it qualifies for.
pub fn route_scene(scene: &SplatScene) -> TierDecision {
    let cpc = coeffs_per_channel(scene);
    let sh_degree = degree_from_coeffs(cpc);

    // Quality tiers engage when there is a non-trivial SH-rest budget, i.e.
    // more than the single DC coefficient per channel.
    let has_sh_rest = cpc > 1;

    let tier = match sh_degree {
        Some(2) | Some(3) => Tier::Full,
        Some(1) => Tier::Partial,
        Some(0) => Tier::Baseline,
        // Non-standard coeff count: if it has SH-rest, route partial so the
        // tiers are not silently disabled; else baseline.
        _ => {
            if has_sh_rest {
                Tier::Partial
            } else {
                Tier::Baseline
            }
        }
    };

    let reason = match tier {
        Tier::Full => format!(
            "SH degree {} detected ({} coeffs/channel): full view-dependent color present \
             — quality tiers (T2.1.R, V5.2) engage the VQ SH-rest palette meaningfully.",
            sh_degree.unwrap_or(0),
            cpc
        ),
        Tier::Partial => format!(
            "SH degree {} detected ({} coeffs/channel): limited view-dependent color \
             — quality tiers engage but with a small SH-rest budget, so gains are muted.",
            sh_degree.map(|d| d.to_string()).unwrap_or_else(|| "?".into()),
            cpc
        ),
        Tier::Baseline => {
            "SH degree 0 detected (DC-only color): this capture has no view-dependent color. \
             Catetus compresses it losslessly via the SF baseline, but the quality tiers \
             (T2.1.R, V5.2) have nothing to re-encode and will be no-ops."
                .to_string()
        }
    };

    TierDecision {
        sh_degree,
        coeffs_per_channel: cpc,
        tier,
        auto_jacobian_effective: has_sh_rest,
        v5_tail_effective: has_sh_rest,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::CoordinateSystem;
    use crate::ir::{Color, Splat, SplatScene, TemporalMode};

    fn scene_with_coeffs_per_channel(cpc: usize) -> SplatScene {
        let color = if cpc <= 1 {
            Color::Rgb([0.1, 0.2, 0.3])
        } else {
            Color::Sh {
                degree: 0, // exercised value is the coeffs length, not this field
                coeffs: vec![0.0_f32; cpc * 3],
            }
        };
        let splat = Splat {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            opacity: 0.5,
            color,
        };
        SplatScene {
            splats: vec![splat],
            coordinate_system: CoordinateSystem::default(),
            semantic_labels: None,
            temporal_mode: TemporalMode::Static,
            lods: None,
            codecgs: None,
        }
    }

    #[test]
    fn rgb_only_routes_to_baseline_with_upsell() {
        let d = route_scene(&scene_with_coeffs_per_channel(1));
        assert_eq!(d.tier, Tier::Baseline);
        assert_eq!(d.sh_degree, Some(0));
        assert!(!d.auto_jacobian_effective);
        assert!(!d.v5_tail_effective);
        assert!(!d.tier.quality_tiers_engage());
    }

    #[test]
    fn sh1_routes_to_partial() {
        let d = route_scene(&scene_with_coeffs_per_channel(4));
        assert_eq!(d.tier, Tier::Partial);
        assert_eq!(d.sh_degree, Some(1));
        assert!(d.auto_jacobian_effective);
    }

    #[test]
    fn sh2_routes_to_full() {
        let d = route_scene(&scene_with_coeffs_per_channel(9));
        assert_eq!(d.tier, Tier::Full);
        assert_eq!(d.sh_degree, Some(2));
        assert!(d.tier.quality_tiers_engage());
    }

    #[test]
    fn sh3_routes_to_full() {
        let d = route_scene(&scene_with_coeffs_per_channel(16));
        assert_eq!(d.tier, Tier::Full);
        assert_eq!(d.sh_degree, Some(3));
        assert!(d.auto_jacobian_effective);
        assert!(d.v5_tail_effective);
    }

    #[test]
    fn empty_scene_routes_to_baseline() {
        let scene = SplatScene {
            splats: vec![],
            coordinate_system: CoordinateSystem::default(),
            semantic_labels: None,
            temporal_mode: TemporalMode::Static,
            lods: None,
            codecgs: None,
        };
        let d = route_scene(&scene);
        assert_eq!(d.tier, Tier::Baseline);
        assert_eq!(d.sh_degree, Some(0));
    }

    #[test]
    fn nonstandard_coeff_count_with_sh_rest_is_partial() {
        let d = route_scene(&scene_with_coeffs_per_channel(7));
        assert_eq!(d.tier, Tier::Partial);
        assert_eq!(d.sh_degree, None);
        assert!(d.auto_jacobian_effective);
    }
}
