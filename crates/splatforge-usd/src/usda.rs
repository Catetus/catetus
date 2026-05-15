//! USDA (text USD) writer and reader.
//!
//! Layout matches SPEC-0011: single `Xform "World"` root, single
//! `ParticleField3DGaussianSplat "Splats"` child carrying the attribute
//! arrays. USDA is the canonical authoring form; USDC ([`crate::write_usdc`])
//! must emit a binary file that decodes to the *same* semantic content
//! after a `usdcat -o out.usda in.usdc` round-trip.

use std::fs;
use std::path::Path;

use splatforge_core::{Color, Splat, SplatScene};

use crate::{UsdError, UsdWriteOpts};

/// Write a USDA (text USD) file.
pub fn write_usda(scene: &SplatScene, path: &Path, opts: &UsdWriteOpts) -> Result<(), UsdError> {
    if scene.splats.is_empty() {
        return Err(UsdError::Malformed("empty scene".to_string()));
    }
    let body = render_usda(scene, opts);
    fs::write(path, body)?;
    Ok(())
}

/// Read a USDA (text USD) file back into the IR.
pub fn read_usda(path: &Path) -> Result<SplatScene, UsdError> {
    let raw = fs::read_to_string(path)?;
    parse_usda(&raw)
}

/// Render the in-memory USDA body. Kept pure so unit tests can assert on
/// the string without writing to disk.
pub fn render_usda(scene: &SplatScene, _opts: &UsdWriteOpts) -> String {
    let n = scene.splats.len();
    let mut out = String::with_capacity(n * 96);

    out.push_str("#usda 1.0\n");
    out.push_str("(\n");
    out.push_str(
        "    doc = \"SplatForge — IR exported as ParticleField3DGaussianSplat. See SPEC-0011.\"\n",
    );
    out.push_str("    upAxis = \"Y\"\n");
    out.push_str("    metersPerUnit = 1\n");
    out.push_str(")\n\n");

    out.push_str("def Xform \"World\"\n{\n");
    out.push_str("    def ParticleField3DGaussianSplat \"Splats\"\n    {\n");

    out.push_str("        point3f[] points = [");
    push_vec3_array(&mut out, n, |i| scene.splats[i].position);
    out.push_str("]\n");

    // USD quaternion convention is (w, x, y, z); IR is (x, y, z, w).
    out.push_str("        quatf[] orientations = [");
    push_quat_array(&mut out, n, |i| {
        let r = scene.splats[i].rotation;
        (r[3], r[0], r[1], r[2])
    });
    out.push_str("]\n");

    out.push_str("        float3[] scales = [");
    push_vec3_array(&mut out, n, |i| scene.splats[i].scale);
    out.push_str("]\n");

    out.push_str("        float[] opacities = [");
    for (i, s) in scene.splats.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_f32(&mut out, s.opacity);
    }
    out.push_str("]\n");

    out.push_str("        color3f[] colorsDC = [");
    push_vec3_array(&mut out, n, |i| match &scene.splats[i].color {
        Color::Rgb(c) => *c,
        Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
    });
    out.push_str("]\n");

    let has_sh = scene
        .splats
        .iter()
        .any(|s| matches!(s.color, Color::Sh { .. }));
    if has_sh {
        out.push_str("        custom float[] splatforge:shCoefficients = [");
        for (i, s) in scene.splats.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            match &s.color {
                Color::Sh { coeffs, .. } => {
                    for (j, c) in coeffs.iter().enumerate() {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        push_f32(&mut out, *c);
                    }
                }
                Color::Rgb(_) => {
                    for j in 0..48 {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        out.push_str("0.0");
                    }
                }
            }
        }
        out.push_str("]\n");
    }

    out.push_str("    }\n}\n");
    out
}

fn push_vec3_array<F>(out: &mut String, n: usize, f: F)
where
    F: Fn(usize) -> [f32; 3],
{
    for i in 0..n {
        if i > 0 {
            out.push_str(", ");
        }
        let v = f(i);
        out.push('(');
        push_f32(out, v[0]);
        out.push_str(", ");
        push_f32(out, v[1]);
        out.push_str(", ");
        push_f32(out, v[2]);
        out.push(')');
    }
}

fn push_quat_array<F>(out: &mut String, n: usize, f: F)
where
    F: Fn(usize) -> (f32, f32, f32, f32),
{
    for i in 0..n {
        if i > 0 {
            out.push_str(", ");
        }
        let (w, x, y, z) = f(i);
        out.push('(');
        push_f32(out, w);
        out.push_str(", ");
        push_f32(out, x);
        out.push_str(", ");
        push_f32(out, y);
        out.push_str(", ");
        push_f32(out, z);
        out.push(')');
    }
}

pub(crate) fn push_f32(out: &mut String, v: f32) {
    if v.fract() == 0.0 && v.is_finite() {
        out.push_str(&format!("{:.1}", v));
    } else {
        out.push_str(&format!("{}", v));
    }
}

/// Best-effort USDA reader. Accepts the canonical layout written by
/// [`write_usda`] *and* the equivalent reformat produced by `usdcat`.
pub fn parse_usda(raw: &str) -> Result<SplatScene, UsdError> {
    let positions = pull_vec3_array(raw, "point3f[] points")?;
    let orientations_wxyz = pull_quat_array(raw, "quatf[] orientations")?;
    let scales = pull_vec3_array(raw, "float3[] scales")?;
    let opacities = pull_scalar_array(raw, "float[] opacities")?;
    let colors = pull_vec3_array(raw, "color3f[] colorsDC")?;

    let n = positions.len();
    if [
        orientations_wxyz.len(),
        scales.len(),
        opacities.len(),
        colors.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(UsdError::Malformed(format!(
            "attribute length mismatch (positions={n})"
        )));
    }

    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        let (w, x, y, z) = (
            orientations_wxyz[i][0],
            orientations_wxyz[i][1],
            orientations_wxyz[i][2],
            orientations_wxyz[i][3],
        );
        splats.push(Splat {
            position: positions[i],
            rotation: [x, y, z, w],
            scale: scales[i],
            opacity: opacities[i],
            color: Color::Rgb(colors[i]),
        });
    }
    let mut scene = SplatScene::new();
    scene.splats = splats;
    Ok(scene)
}

fn pull_vec3_array(raw: &str, key: &str) -> Result<Vec<[f32; 3]>, UsdError> {
    let body = pull_array_body(raw, key)?;
    let mut out = Vec::new();
    for triple in split_parens(&body) {
        let parts: Vec<&str> = triple.split(',').map(str::trim).collect();
        if parts.len() != 3 {
            return Err(UsdError::Malformed(format!(
                "{key}: expected 3-tuple, got {triple:?}"
            )));
        }
        let a = parts[0]
            .parse::<f32>()
            .map_err(|e| UsdError::Malformed(format!("{key}: {e}")))?;
        let b = parts[1]
            .parse::<f32>()
            .map_err(|e| UsdError::Malformed(format!("{key}: {e}")))?;
        let c = parts[2]
            .parse::<f32>()
            .map_err(|e| UsdError::Malformed(format!("{key}: {e}")))?;
        out.push([a, b, c]);
    }
    Ok(out)
}

fn pull_quat_array(raw: &str, key: &str) -> Result<Vec<[f32; 4]>, UsdError> {
    let body = pull_array_body(raw, key)?;
    let mut out = Vec::new();
    for tuple in split_parens(&body) {
        let parts: Vec<&str> = tuple.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(UsdError::Malformed(format!(
                "{key}: expected 4-tuple, got {tuple:?}"
            )));
        }
        let mut q = [0.0f32; 4];
        for (i, p) in parts.iter().enumerate() {
            q[i] = p
                .parse::<f32>()
                .map_err(|e| UsdError::Malformed(format!("{key}: {e}")))?;
        }
        out.push(q);
    }
    Ok(out)
}

fn pull_scalar_array(raw: &str, key: &str) -> Result<Vec<f32>, UsdError> {
    let body = pull_array_body(raw, key)?;
    let mut out = Vec::new();
    for tok in body.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        out.push(
            tok.parse::<f32>()
                .map_err(|e| UsdError::Malformed(format!("{key}: {e}")))?,
        );
    }
    Ok(out)
}

fn pull_array_body(raw: &str, key: &str) -> Result<String, UsdError> {
    let idx = raw
        .find(key)
        .ok_or_else(|| UsdError::Malformed(format!("missing attribute: {key}")))?;
    let after = &raw[idx + key.len()..];
    let eq = after
        .find('=')
        .ok_or_else(|| UsdError::Malformed(format!("{key}: no '=' after key")))?;
    let lb = after[eq..]
        .find('[')
        .ok_or_else(|| UsdError::Malformed(format!("{key}: no '[' after '='")))?;
    let start = eq + lb + 1;
    let rb = after[start..]
        .find(']')
        .ok_or_else(|| UsdError::Malformed(format!("{key}: no closing ']'")))?;
    Ok(after[start..start + rb].to_string())
}

fn split_parens(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0;
    let mut buf = String::new();
    for ch in body.chars() {
        match ch {
            '(' => {
                depth += 1;
                if depth > 1 {
                    buf.push(ch);
                }
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    if !buf.trim().is_empty() {
                        out.push(buf.trim().to_string());
                    }
                    buf.clear();
                } else {
                    buf.push(ch);
                }
            }
            _ if depth > 0 => buf.push(ch),
            _ => {}
        }
    }
    out
}
