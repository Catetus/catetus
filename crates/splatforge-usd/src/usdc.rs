//! USDC (Pixar Crate binary, version 0.0.1) writer and reader.
//!
//! ## Why version 0.0.1?
//!
//! Pixar's `SdfFileVersion::CanRead` returns true when the file's major matches
//! the software's major and the file's minor is `<=` the software's minor.
//! That means any minor version `<= software.minor` is forward-compatible. We
//! target the *oldest* schema (0.0.1) because:
//!
//! * Fields, FieldSets, Paths, and Specs are all written *uncompressed* — no
//!   integer-coder needed. (The integer coder is a 70-byte VLE scheme that
//!   Pixar tuned for typical authoring sizes; reimplementing it bit-exactly
//!   is a 200-line side project.)
//! * Arrays use the simple `[u32 shape=1, u32 size, T data...]` layout
//!   (version `< 0.5.0`) — no LZ4 on int32/int64/float arrays.
//! * Only the TOKENS section uses LZ4 (`TfFastCompression`, single chunk).
//!
//! ## Format
//!
//! Source of truth: `pxr/usd/sdf/crateFile.{h,cpp}` in OpenUSD release branch.
//!
//! ```text
//! Bootstrap (88 bytes at offset 0):
//!   [8 ]  "PXR-USDC"
//!   [8 ]  version[8] = { major=0, minor=0, patch=1, 0,0,0,0,0 }
//!   [8 ]  int64 tocOffset
//!   [64]  reserved (zero)
//!
//! Then sections in stream order: TOKENS, STRINGS, FIELDS, FIELDSETS,
//! PATHS, SPECS. Each section is just raw bytes; the TOC at the end records
//! (name, start, size).
//!
//! TOC at tocOffset:
//!   u64 numSections
//!   numSections * { u8 name[16], i64 start, i64 size }
//! ```
//!
//! See `SPEC-GAPS.md` for clauses where OpenUSD 26.03's
//! `ParticleField3DGaussianSplat` schema is under-specified.

use std::fs;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::Path;

use splatforge_core::{Color, Splat, SplatScene};

use crate::{UsdError, UsdWriteOpts};

// ----------------------- Bootstrap constants ---------------------------

const MAGIC: [u8; 8] = *b"PXR-USDC";
const VERSION: [u8; 8] = [0, 0, 1, 0, 0, 0, 0, 0];
const BOOTSTRAP_SIZE: usize = 88;
const SECTION_NAME_LEN: usize = 16;

const SEC_TOKENS: &[u8] = b"TOKENS";
const SEC_STRINGS: &[u8] = b"STRINGS";
const SEC_FIELDS: &[u8] = b"FIELDS";
const SEC_FIELDSETS: &[u8] = b"FIELDSETS";
const SEC_PATHS: &[u8] = b"PATHS";
const SEC_SPECS: &[u8] = b"SPECS";

// ----------------------- TypeEnum (crateDataTypes.h) -------------------

#[allow(dead_code)]
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum TypeEnum {
    Invalid = 0,
    Bool = 1,
    Float = 8,
    Double = 9,
    Token = 11,
    Quatf = 17,
    Vec3f = 24,
    TokenVector = 41,
    Specifier = 42,
    Permission = 43,
    Variability = 44,
}

// ----------------------- ValueRep bits ---------------------------------

const IS_ARRAY_BIT: u64 = 1u64 << 63;
const IS_INLINED_BIT: u64 = 1u64 << 62;
#[allow(dead_code)]
const IS_COMPRESSED_BIT: u64 = 1u64 << 61;
const PAYLOAD_MASK: u64 = (1u64 << 48) - 1;
const TYPE_SHIFT: u32 = 48;

fn value_rep(ty: TypeEnum, is_inlined: bool, is_array: bool, payload: u64) -> u64 {
    (if is_array { IS_ARRAY_BIT } else { 0 })
        | (if is_inlined { IS_INLINED_BIT } else { 0 })
        | ((ty as u64) << TYPE_SHIFT)
        | (payload & PAYLOAD_MASK)
}

fn vr_type(rep: u64) -> u8 {
    ((rep >> TYPE_SHIFT) & 0xff) as u8
}
fn vr_is_array(rep: u64) -> bool {
    (rep & IS_ARRAY_BIT) != 0
}
fn vr_is_inlined(rep: u64) -> bool {
    (rep & IS_INLINED_BIT) != 0
}
fn vr_payload(rep: u64) -> u64 {
    rep & PAYLOAD_MASK
}

// ----------------------- SdfSpecType, Specifier, Variability ------------

#[allow(dead_code)]
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SpecType {
    Unknown = 0,
    Attribute = 1,
    Connection = 2,
    Expression = 3,
    Mapper = 4,
    MapperArg = 5,
    Prim = 6,
    PseudoRoot = 7,
    Relationship = 8,
    RelationshipTarget = 9,
    Variant = 10,
    VariantSet = 11,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Specifier {
    Def = 0,
    #[allow(dead_code)]
    Over = 1,
    #[allow(dead_code)]
    Class = 2,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Variability {
    Varying = 0,
    #[allow(dead_code)]
    Uniform = 1,
}

// ----------------------- Field / Path / Spec records -------------------

#[derive(Clone, Debug)]
struct Field {
    token_index: u32,
    value_rep: u64,
}

#[derive(Clone, Debug)]
struct PathNode {
    path_index: u32,
    element_token_index: u32,
    is_prim_property: bool,
    children: Vec<PathNode>,
}

#[derive(Clone, Debug)]
struct SpecRec {
    path_index: u32,
    field_set_index: u32,
    spec_type: SpecType,
}

// ----------------------- Token table -----------------------------------

/// Deterministic token table. Insertion-ordered. Index 0 is always `";-)"`
/// (Pixar's path-coder workaround, see `crateFile.cpp` StartPacking()).
struct TokenTable {
    by_str: std::collections::HashMap<String, u32>,
    ordered: Vec<String>,
}

impl TokenTable {
    fn new() -> Self {
        let mut t = Self {
            by_str: std::collections::HashMap::new(),
            ordered: Vec::new(),
        };
        t.intern(";-)");
        t
    }
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.by_str.get(s) {
            return i;
        }
        let i = self.ordered.len() as u32;
        self.ordered.push(s.to_string());
        self.by_str.insert(s.to_string(), i);
        i
    }
}

// ----------------------- Field / FieldSet dedup ------------------------

#[derive(Default)]
struct FieldTable {
    by_pair: std::collections::HashMap<(u32, u64), u32>,
    ordered: Vec<Field>,
}
impl FieldTable {
    fn intern(&mut self, token_index: u32, value_rep: u64) -> u32 {
        if let Some(&i) = self.by_pair.get(&(token_index, value_rep)) {
            return i;
        }
        let i = self.ordered.len() as u32;
        self.ordered.push(Field {
            token_index,
            value_rep,
        });
        self.by_pair.insert((token_index, value_rep), i);
        i
    }
}

#[derive(Default)]
struct FieldSetTable {
    stream: Vec<u32>,
    by_seq: std::collections::HashMap<Vec<u32>, u32>,
}
impl FieldSetTable {
    fn intern(&mut self, fields: &[u32]) -> u32 {
        if let Some(&i) = self.by_seq.get(fields) {
            return i;
        }
        let i = self.stream.len() as u32;
        self.stream.extend_from_slice(fields);
        self.stream.push(u32::MAX);
        self.by_seq.insert(fields.to_vec(), i);
        i
    }
}

// ============================================================
//                          WRITER
// ============================================================

/// Write a USDC (Pixar Crate binary) file at version 0.0.1.
pub fn write_usdc(scene: &SplatScene, path: &Path, opts: &UsdWriteOpts) -> Result<(), UsdError> {
    if scene.splats.is_empty() {
        return Err(UsdError::Malformed("empty scene".to_string()));
    }
    let bytes = encode_usdc(scene, opts)?;
    fs::write(path, bytes)?;
    Ok(())
}

/// In-memory USDC encoder. Pure so tests can inspect the byte stream.
pub(crate) fn encode_usdc(scene: &SplatScene, _opts: &UsdWriteOpts) -> Result<Vec<u8>, UsdError> {
    let mut tokens = TokenTable::new();
    let mut fields = FieldTable::default();
    let mut field_sets = FieldSetTable::default();

    // Value-data buffer (relative offsets; rebased to absolute by `+ BOOTSTRAP_SIZE`).
    let mut values = Cursor::new(Vec::<u8>::new());

    // ---- field-name tokens ----
    let tok_prim_children = tokens.intern("primChildren");
    let tok_specifier = tokens.intern("specifier");
    let tok_type_name = tokens.intern("typeName");
    let tok_properties = tokens.intern("properties");
    let tok_custom = tokens.intern("custom");
    let tok_default = tokens.intern("default");
    let tok_variability = tokens.intern("variability");

    // ---- value tokens ----
    let tok_world = tokens.intern("World");
    let tok_splats = tokens.intern("Splats");
    let tok_xform = tokens.intern("Xform");
    let tok_particle_field = tokens.intern("ParticleField3DGaussianSplat");
    let tok_point3f_arr = tokens.intern("point3f[]");
    let tok_quatf_arr = tokens.intern("quatf[]");
    let tok_float3_arr = tokens.intern("float3[]");
    let tok_float_arr = tokens.intern("float[]");
    let tok_color3f_arr = tokens.intern("color3f[]");

    // ---- property name tokens ----
    let tok_points = tokens.intern("points");
    let tok_orientations = tokens.intern("orientations");
    let tok_scales = tokens.intern("scales");
    let tok_opacities = tokens.intern("opacities");
    let tok_colors_dc = tokens.intern("colorsDC");

    let has_sh = scene
        .splats
        .iter()
        .any(|s| matches!(s.color, Color::Sh { .. }));
    let tok_sh_coeffs = if has_sh {
        Some(tokens.intern("splatforge:shCoefficients"))
    } else {
        None
    };

    // ---- gather IR arrays ----
    let positions: Vec<[f32; 3]> = scene.splats.iter().map(|s| s.position).collect();
    let orientations_wxyz: Vec<[f32; 4]> = scene.splats.iter().map(|s| s.rotation).collect();
    let scales: Vec<[f32; 3]> = scene.splats.iter().map(|s| s.scale).collect();
    let opacities: Vec<f32> = scene.splats.iter().map(|s| s.opacity).collect();
    let colors_dc: Vec<[f32; 3]> = scene
        .splats
        .iter()
        .map(|s| match &s.color {
            Color::Rgb(c) => *c,
            Color::Sh { coeffs, .. } => [coeffs[0], coeffs[1], coeffs[2]],
        })
        .collect();
    let sh_packed: Option<Vec<f32>> = if has_sh {
        let mut packed = Vec::with_capacity(scene.splats.len() * 48);
        for s in &scene.splats {
            match &s.color {
                Color::Sh { coeffs, .. } => packed.extend_from_slice(coeffs),
                Color::Rgb(_) => packed.extend(std::iter::repeat_n(0.0, 48)),
            }
        }
        Some(packed)
    } else {
        None
    };

    let off_points = write_array_vec3f(&mut values, &positions)?;
    let off_orient = write_array_quatf_from_wxyz(&mut values, &orientations_wxyz)?;
    let off_scales = write_array_vec3f(&mut values, &scales)?;
    let off_opac = write_array_f32(&mut values, &opacities)?;
    let off_colors = write_array_vec3f(&mut values, &colors_dc)?;
    let off_sh = match &sh_packed {
        Some(d) => Some(write_array_f32(&mut values, d)?),
        None => None,
    };

    let rep_points = value_rep(
        TypeEnum::Vec3f,
        false,
        true,
        BOOTSTRAP_SIZE as u64 + off_points,
    );
    let rep_orient = value_rep(
        TypeEnum::Quatf,
        false,
        true,
        BOOTSTRAP_SIZE as u64 + off_orient,
    );
    let rep_scales = value_rep(
        TypeEnum::Vec3f,
        false,
        true,
        BOOTSTRAP_SIZE as u64 + off_scales,
    );
    let rep_opac = value_rep(
        TypeEnum::Float,
        false,
        true,
        BOOTSTRAP_SIZE as u64 + off_opac,
    );
    let rep_colors = value_rep(
        TypeEnum::Vec3f,
        false,
        true,
        BOOTSTRAP_SIZE as u64 + off_colors,
    );
    let rep_sh = off_sh.map(|o| value_rep(TypeEnum::Float, false, true, BOOTSTRAP_SIZE as u64 + o));

    // typeName Tokens (inlined).
    let rep_tn_point3f = value_rep(TypeEnum::Token, true, false, tok_point3f_arr as u64);
    let rep_tn_quatf = value_rep(TypeEnum::Token, true, false, tok_quatf_arr as u64);
    let rep_tn_float3 = value_rep(TypeEnum::Token, true, false, tok_float3_arr as u64);
    let rep_tn_float = value_rep(TypeEnum::Token, true, false, tok_float_arr as u64);
    let rep_tn_color3f = value_rep(TypeEnum::Token, true, false, tok_color3f_arr as u64);

    let rep_spec_def = value_rep(TypeEnum::Specifier, true, false, Specifier::Def as u64);
    let rep_var_varying = value_rep(
        TypeEnum::Variability,
        true,
        false,
        Variability::Varying as u64,
    );
    let rep_bool_true = value_rep(TypeEnum::Bool, true, false, 1);

    let rep_tn_xform = value_rep(TypeEnum::Token, true, false, tok_xform as u64);
    let rep_tn_particle = value_rep(TypeEnum::Token, true, false, tok_particle_field as u64);

    let off_root_children = write_token_vector(&mut values, &[tok_world])?;
    let off_world_children = write_token_vector(&mut values, &[tok_splats])?;
    let mut prop_toks = vec![
        tok_points,
        tok_orientations,
        tok_scales,
        tok_opacities,
        tok_colors_dc,
    ];
    if let Some(t) = tok_sh_coeffs {
        prop_toks.push(t);
    }
    let off_splats_props = write_token_vector(&mut values, &prop_toks)?;

    let rep_root_children = value_rep(
        TypeEnum::TokenVector,
        false,
        false,
        BOOTSTRAP_SIZE as u64 + off_root_children,
    );
    let rep_world_children = value_rep(
        TypeEnum::TokenVector,
        false,
        false,
        BOOTSTRAP_SIZE as u64 + off_world_children,
    );
    let rep_splats_props = value_rep(
        TypeEnum::TokenVector,
        false,
        false,
        BOOTSTRAP_SIZE as u64 + off_splats_props,
    );

    // ---- Field sets ----
    let f_root_primchildren = fields.intern(tok_prim_children, rep_root_children);
    let fs_pseudoroot = field_sets.intern(&[f_root_primchildren]);

    let f_world_primchildren = fields.intern(tok_prim_children, rep_world_children);
    let f_specifier_def = fields.intern(tok_specifier, rep_spec_def);
    let f_world_typename = fields.intern(tok_type_name, rep_tn_xform);
    let fs_world = field_sets.intern(&[f_world_primchildren, f_specifier_def, f_world_typename]);

    let f_splats_props = fields.intern(tok_properties, rep_splats_props);
    let f_splats_typename = fields.intern(tok_type_name, rep_tn_particle);
    let fs_splats = field_sets.intern(&[f_splats_props, f_specifier_def, f_splats_typename]);

    let build_attr_fields = |type_name_rep: u64,
                             default_rep: u64,
                             custom: bool,
                             fields: &mut FieldTable,
                             field_sets: &mut FieldSetTable|
     -> u32 {
        let f_tn = fields.intern(tok_type_name, type_name_rep);
        let f_default = fields.intern(tok_default, default_rep);
        let f_variability = fields.intern(tok_variability, rep_var_varying);
        if custom {
            let f_custom = fields.intern(tok_custom, rep_bool_true);
            field_sets.intern(&[f_tn, f_default, f_custom, f_variability])
        } else {
            field_sets.intern(&[f_tn, f_default, f_variability])
        }
    };

    let fs_points = build_attr_fields(
        rep_tn_point3f,
        rep_points,
        false,
        &mut fields,
        &mut field_sets,
    );
    let fs_orient = build_attr_fields(
        rep_tn_quatf,
        rep_orient,
        false,
        &mut fields,
        &mut field_sets,
    );
    let fs_scales = build_attr_fields(
        rep_tn_float3,
        rep_scales,
        false,
        &mut fields,
        &mut field_sets,
    );
    let fs_opac = build_attr_fields(rep_tn_float, rep_opac, false, &mut fields, &mut field_sets);
    let fs_colors = build_attr_fields(
        rep_tn_color3f,
        rep_colors,
        false,
        &mut fields,
        &mut field_sets,
    );
    let fs_sh =
        rep_sh.map(|r| build_attr_fields(rep_tn_float, r, true, &mut fields, &mut field_sets));

    // ---- Path tree ----
    let mut next_path_idx: u32 = 0;
    let mut take_idx = || -> u32 {
        let i = next_path_idx;
        next_path_idx += 1;
        i
    };
    let pi_root = take_idx();
    let pi_world = take_idx();
    let pi_splats = take_idx();
    let pi_points = take_idx();
    let pi_orient = take_idx();
    let pi_scales = take_idx();
    let pi_opac = take_idx();
    let pi_colors = take_idx();
    let pi_sh = if tok_sh_coeffs.is_some() {
        Some(take_idx())
    } else {
        None
    };

    let mut splat_prop_children = vec![
        PathNode {
            path_index: pi_points,
            element_token_index: tok_points,
            is_prim_property: true,
            children: vec![],
        },
        PathNode {
            path_index: pi_orient,
            element_token_index: tok_orientations,
            is_prim_property: true,
            children: vec![],
        },
        PathNode {
            path_index: pi_scales,
            element_token_index: tok_scales,
            is_prim_property: true,
            children: vec![],
        },
        PathNode {
            path_index: pi_opac,
            element_token_index: tok_opacities,
            is_prim_property: true,
            children: vec![],
        },
        PathNode {
            path_index: pi_colors,
            element_token_index: tok_colors_dc,
            is_prim_property: true,
            children: vec![],
        },
    ];
    if let (Some(t), Some(pi)) = (tok_sh_coeffs, pi_sh) {
        splat_prop_children.push(PathNode {
            path_index: pi,
            element_token_index: t,
            is_prim_property: true,
            children: vec![],
        });
    }

    let root_node = PathNode {
        path_index: pi_root,
        // PseudoRoot's element token is the ";-)" sentinel at index 0.
        element_token_index: 0,
        is_prim_property: false,
        children: vec![PathNode {
            path_index: pi_world,
            element_token_index: tok_world,
            is_prim_property: false,
            children: vec![PathNode {
                path_index: pi_splats,
                element_token_index: tok_splats,
                is_prim_property: false,
                children: splat_prop_children,
            }],
        }],
    };

    // ---- Spec records ----
    let mut specs = vec![
        SpecRec {
            path_index: pi_root,
            field_set_index: fs_pseudoroot,
            spec_type: SpecType::PseudoRoot,
        },
        SpecRec {
            path_index: pi_world,
            field_set_index: fs_world,
            spec_type: SpecType::Prim,
        },
        SpecRec {
            path_index: pi_splats,
            field_set_index: fs_splats,
            spec_type: SpecType::Prim,
        },
        SpecRec {
            path_index: pi_points,
            field_set_index: fs_points,
            spec_type: SpecType::Attribute,
        },
        SpecRec {
            path_index: pi_orient,
            field_set_index: fs_orient,
            spec_type: SpecType::Attribute,
        },
        SpecRec {
            path_index: pi_scales,
            field_set_index: fs_scales,
            spec_type: SpecType::Attribute,
        },
        SpecRec {
            path_index: pi_opac,
            field_set_index: fs_opac,
            spec_type: SpecType::Attribute,
        },
        SpecRec {
            path_index: pi_colors,
            field_set_index: fs_colors,
            spec_type: SpecType::Attribute,
        },
    ];
    if let (Some(pi), Some(fs)) = (pi_sh, fs_sh) {
        specs.push(SpecRec {
            path_index: pi,
            field_set_index: fs,
            spec_type: SpecType::Attribute,
        });
    }

    // ---- Lay out the file ----
    let mut out = Cursor::new(Vec::<u8>::new());

    out.write_all(&[0u8; BOOTSTRAP_SIZE])?;
    out.write_all(&values.into_inner())?;

    let tok_start = out.position();
    write_tokens(&mut out, &tokens.ordered)?;
    let tok_size = out.position() - tok_start;

    let str_start = out.position();
    out.write_all(&0u64.to_le_bytes())?; // numStrings = 0
    let str_size = out.position() - str_start;

    let fields_start = out.position();
    out.write_all(&(fields.ordered.len() as u64).to_le_bytes())?;
    for f in &fields.ordered {
        // u32 padding, u32 tokenIndex, u64 valueRep.
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&f.token_index.to_le_bytes())?;
        out.write_all(&f.value_rep.to_le_bytes())?;
    }
    let fields_size = out.position() - fields_start;

    let fset_start = out.position();
    out.write_all(&(field_sets.stream.len() as u64).to_le_bytes())?;
    for &v in &field_sets.stream {
        out.write_all(&v.to_le_bytes())?;
    }
    let fset_size = out.position() - fset_start;

    let paths_start = out.position();
    out.write_all(&(next_path_idx as u64).to_le_bytes())?;
    write_path_tree(&mut out, &root_node)?;
    let paths_size = out.position() - paths_start;

    let specs_start = out.position();
    out.write_all(&(specs.len() as u64).to_le_bytes())?;
    for s in &specs {
        // u32 padding, u32 pathIndex, u32 fieldSetIndex, u32 specType.
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&s.path_index.to_le_bytes())?;
        out.write_all(&s.field_set_index.to_le_bytes())?;
        out.write_all(&(s.spec_type as u32).to_le_bytes())?;
    }
    let specs_size = out.position() - specs_start;

    // TOC.
    let toc_offset = out.position();
    out.write_all(&6u64.to_le_bytes())?;
    for (name, start, size) in [
        (SEC_TOKENS, tok_start, tok_size),
        (SEC_STRINGS, str_start, str_size),
        (SEC_FIELDS, fields_start, fields_size),
        (SEC_FIELDSETS, fset_start, fset_size),
        (SEC_PATHS, paths_start, paths_size),
        (SEC_SPECS, specs_start, specs_size),
    ] {
        let mut nb = [0u8; SECTION_NAME_LEN];
        nb[..name.len()].copy_from_slice(name);
        out.write_all(&nb)?;
        out.write_all(&(start as i64).to_le_bytes())?;
        out.write_all(&(size as i64).to_le_bytes())?;
    }

    // Rewrite the bootstrap with the real TOC offset.
    out.seek(SeekFrom::Start(0))?;
    out.write_all(&MAGIC)?;
    out.write_all(&VERSION)?;
    out.write_all(&(toc_offset as i64).to_le_bytes())?;
    out.write_all(&[0u8; 64])?;

    Ok(out.into_inner())
}

fn align8(c: &mut Cursor<Vec<u8>>) -> std::io::Result<()> {
    let pos = c.position();
    let pad = (8 - (pos % 8)) % 8;
    if pad > 0 {
        let zeros = [0u8; 8];
        c.write_all(&zeros[..pad as usize])?;
    }
    Ok(())
}

fn write_array_f32(values: &mut Cursor<Vec<u8>>, data: &[f32]) -> std::io::Result<u64> {
    align8(values)?;
    let off = values.position();
    values.write_all(&1u32.to_le_bytes())?;
    values.write_all(&(data.len() as u32).to_le_bytes())?;
    for v in data {
        values.write_all(&v.to_le_bytes())?;
    }
    Ok(off)
}

fn write_array_vec3f(values: &mut Cursor<Vec<u8>>, data: &[[f32; 3]]) -> std::io::Result<u64> {
    align8(values)?;
    let off = values.position();
    values.write_all(&1u32.to_le_bytes())?;
    values.write_all(&(data.len() as u32).to_le_bytes())?;
    for v in data {
        for c in v {
            values.write_all(&c.to_le_bytes())?;
        }
    }
    Ok(off)
}

/// Pixar's `GfQuatf` wire layout is imaginary then real: `(x, y, z, w)`.
/// Input here is `(x, y, z, w)` (IR order); on-disk also `(x, y, z, w)`.
fn write_array_quatf_from_wxyz(
    values: &mut Cursor<Vec<u8>>,
    data: &[[f32; 4]],
) -> std::io::Result<u64> {
    align8(values)?;
    let off = values.position();
    values.write_all(&1u32.to_le_bytes())?;
    values.write_all(&(data.len() as u32).to_le_bytes())?;
    for q in data {
        // q is (x, y, z, w) in IR (`splatforge_core::Splat::rotation`).
        // On-disk layout matches: x, y, z, w.
        values.write_all(&q[0].to_le_bytes())?;
        values.write_all(&q[1].to_le_bytes())?;
        values.write_all(&q[2].to_le_bytes())?;
        values.write_all(&q[3].to_le_bytes())?;
    }
    Ok(off)
}

fn write_token_vector(values: &mut Cursor<Vec<u8>>, toks: &[u32]) -> std::io::Result<u64> {
    align8(values)?;
    let off = values.position();
    values.write_all(&(toks.len() as u64).to_le_bytes())?;
    for t in toks {
        values.write_all(&t.to_le_bytes())?;
    }
    Ok(off)
}

fn write_tokens(out: &mut Cursor<Vec<u8>>, ordered: &[String]) -> Result<(), UsdError> {
    // Version 0.0.1 uses *uncompressed* tokens. Layout:
    //   u64 numTokens
    //   u64 totalBytes
    //   <totalBytes> null-terminated UTF-8 strings (last byte must be NUL)
    //
    // The LZ4-compressed form is a 0.4.0+ feature.
    let mut raw = Vec::<u8>::new();
    for s in ordered {
        raw.extend_from_slice(s.as_bytes());
        raw.push(0);
    }
    out.write_all(&(ordered.len() as u64).to_le_bytes())?;
    out.write_all(&(raw.len() as u64).to_le_bytes())?;
    out.write_all(&raw)?;
    Ok(())
}

fn write_path_tree(out: &mut Cursor<Vec<u8>>, root: &PathNode) -> std::io::Result<()> {
    write_path_list(out, std::slice::from_ref(root))
}

fn write_path_list(out: &mut Cursor<Vec<u8>>, siblings: &[PathNode]) -> std::io::Result<()> {
    for (i, node) in siblings.iter().enumerate() {
        let has_child = !node.children.is_empty();
        let has_sibling = i + 1 < siblings.len();
        let mut bits = 0u8;
        if has_child {
            bits |= 0b001;
        }
        if has_sibling {
            bits |= 0b010;
        }
        if node.is_prim_property {
            bits |= 0b100;
        }

        // _PathItemHeader_0_0_1: u32 _pad, u32 pathIndex, u32 elementTokenIndex,
        // u8 bits + 3 bytes trailing pad (to 16-byte struct alignment).
        out.write_all(&0u32.to_le_bytes())?;
        out.write_all(&node.path_index.to_le_bytes())?;
        out.write_all(&node.element_token_index.to_le_bytes())?;
        out.write_all(&[bits, 0, 0, 0])?;

        let sibling_ptr_offset = if has_child && has_sibling {
            let pos = out.position();
            out.write_all(&(-1i64).to_le_bytes())?;
            Some(pos)
        } else {
            None
        };

        if has_child {
            write_path_list(out, &node.children)?;
        }

        if let Some(off) = sibling_ptr_offset {
            let cur = out.position();
            out.seek(SeekFrom::Start(off))?;
            out.write_all(&(cur as i64).to_le_bytes())?;
            out.seek(SeekFrom::Start(cur))?;
        }
    }
    Ok(())
}

// ============================================================
//                          READER
// ============================================================

/// Read a USDC binary file and return the IR scene.
///
/// Supports the subset emitted by [`write_usdc`]: version 0.0.1, our
/// `ParticleField3DGaussianSplat` schema. Useful for in-process round-trip
/// tests; for production "consume any USDC", call out to `usdcat -o tmp.usda`
/// and use [`crate::read_usda`].
pub fn read_usdc(path: &Path) -> Result<SplatScene, UsdError> {
    let bytes = fs::read(path)?;
    decode_usdc(&bytes)
}

pub(crate) fn decode_usdc(bytes: &[u8]) -> Result<SplatScene, UsdError> {
    if bytes.len() < BOOTSTRAP_SIZE {
        return Err(UsdError::MalformedUsdc(
            "file shorter than bootstrap".into(),
        ));
    }
    if bytes[..8] != MAGIC {
        return Err(UsdError::MalformedUsdc("bad magic".into()));
    }
    let (maj, min, patch) = (bytes[8], bytes[9], bytes[10]);
    if maj != 0 || min != 0 {
        return Err(UsdError::UnsupportedUsdcVersion(maj, min, patch));
    }
    let toc_offset = i64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;

    if toc_offset + 8 > bytes.len() {
        return Err(UsdError::MalformedUsdc("toc offset out of range".into()));
    }
    let n_sections =
        u64::from_le_bytes(bytes[toc_offset..toc_offset + 8].try_into().unwrap()) as usize;
    let mut sections = std::collections::HashMap::<String, (usize, usize)>::new();
    let mut p = toc_offset + 8;
    for _ in 0..n_sections {
        if p + 32 > bytes.len() {
            return Err(UsdError::MalformedUsdc("toc truncated".into()));
        }
        let name_raw = &bytes[p..p + 16];
        let name_end = name_raw.iter().position(|&b| b == 0).unwrap_or(16);
        let name = std::str::from_utf8(&name_raw[..name_end])
            .map_err(|e| UsdError::MalformedUsdc(format!("section name: {e}")))?
            .to_string();
        let start = i64::from_le_bytes(bytes[p + 16..p + 24].try_into().unwrap()) as usize;
        let size = i64::from_le_bytes(bytes[p + 24..p + 32].try_into().unwrap()) as usize;
        sections.insert(name, (start, size));
        p += 32;
    }

    let (ts, tz) = sections
        .get("TOKENS")
        .ok_or_else(|| UsdError::MalformedUsdc("missing TOKENS section".into()))?;
    let tokens = read_tokens(&bytes[*ts..*ts + *tz])?;

    let (fs_, fz) = sections
        .get("FIELDS")
        .ok_or_else(|| UsdError::MalformedUsdc("missing FIELDS section".into()))?;
    let field_bytes = &bytes[*fs_..*fs_ + *fz];
    let n_fields = u64::from_le_bytes(field_bytes[..8].try_into().unwrap()) as usize;
    let mut fields_vec = Vec::with_capacity(n_fields);
    let mut cursor = 8;
    for _ in 0..n_fields {
        if cursor + 16 > field_bytes.len() {
            return Err(UsdError::MalformedUsdc("fields truncated".into()));
        }
        let token_index =
            u32::from_le_bytes(field_bytes[cursor + 4..cursor + 8].try_into().unwrap());
        let value_rep =
            u64::from_le_bytes(field_bytes[cursor + 8..cursor + 16].try_into().unwrap());
        fields_vec.push(Field {
            token_index,
            value_rep,
        });
        cursor += 16;
    }

    let (fss, fsz) = sections
        .get("FIELDSETS")
        .ok_or_else(|| UsdError::MalformedUsdc("missing FIELDSETS section".into()))?;
    let fs_bytes = &bytes[*fss..*fss + *fsz];
    let n_fset = u64::from_le_bytes(fs_bytes[..8].try_into().unwrap()) as usize;
    let mut fset = Vec::with_capacity(n_fset);
    for i in 0..n_fset {
        let off = 8 + i * 4;
        fset.push(u32::from_le_bytes(
            fs_bytes[off..off + 4].try_into().unwrap(),
        ));
    }

    let (ss, sz) = sections
        .get("SPECS")
        .ok_or_else(|| UsdError::MalformedUsdc("missing SPECS section".into()))?;
    let spec_bytes = &bytes[*ss..*ss + *sz];
    let n_specs = u64::from_le_bytes(spec_bytes[..8].try_into().unwrap()) as usize;
    let mut specs = Vec::with_capacity(n_specs);
    let mut sc = 8;
    for _ in 0..n_specs {
        if sc + 16 > spec_bytes.len() {
            return Err(UsdError::MalformedUsdc("specs truncated".into()));
        }
        let path_index = u32::from_le_bytes(spec_bytes[sc + 4..sc + 8].try_into().unwrap());
        let field_set_index = u32::from_le_bytes(spec_bytes[sc + 8..sc + 12].try_into().unwrap());
        let spec_type = u32::from_le_bytes(spec_bytes[sc + 12..sc + 16].try_into().unwrap());
        specs.push((path_index, field_set_index, spec_type));
        sc += 16;
    }

    // For each Attribute spec, find its `default` array based on `typeName`.
    let want_tn = |s: &str| -> Option<u32> { tokens.iter().position(|t| t == s).map(|i| i as u32) };
    let t_default = want_tn("default");
    let t_typename = want_tn("typeName");

    let mut points: Option<Vec<[f32; 3]>> = None;
    let mut orient: Option<Vec<[f32; 4]>> = None;
    let mut scales_o: Option<Vec<[f32; 3]>> = None;
    let mut opac_o: Option<Vec<f32>> = None;
    let mut colors_o: Option<Vec<[f32; 3]>> = None;
    let mut sh_o: Option<Vec<f32>> = None;

    for (_pi, fsi, st) in &specs {
        if *st != SpecType::Attribute as u32 {
            continue;
        }
        let mut field_indices = Vec::new();
        let mut i = *fsi as usize;
        while i < fset.len() {
            let v = fset[i];
            if v == u32::MAX {
                break;
            }
            field_indices.push(v);
            i += 1;
        }
        let mut type_name_tok: Option<u32> = None;
        let mut default_rep: Option<u64> = None;
        for fi in &field_indices {
            let f = &fields_vec[*fi as usize];
            if Some(f.token_index) == t_typename && vr_is_inlined(f.value_rep) {
                type_name_tok = Some(vr_payload(f.value_rep) as u32);
            } else if Some(f.token_index) == t_default {
                default_rep = Some(f.value_rep);
            }
        }
        let (Some(tn), Some(rep)) = (type_name_tok, default_rep) else {
            continue;
        };
        let tn_str = tokens
            .get(tn as usize)
            .map(String::as_str)
            .unwrap_or_default();
        if !vr_is_array(rep) {
            continue;
        }
        let payload = vr_payload(rep) as usize;
        match tn_str {
            "point3f[]" => {
                points = Some(read_array_vec3f(bytes, payload)?);
            }
            "color3f[]" => {
                colors_o = Some(read_array_vec3f(bytes, payload)?);
            }
            "float3[]" => {
                scales_o = Some(read_array_vec3f(bytes, payload)?);
            }
            "quatf[]" => {
                orient = Some(read_array_quatf(bytes, payload)?);
            }
            "float[]" => {
                let arr = read_array_f32(bytes, payload)?;
                if opac_o.is_none() {
                    opac_o = Some(arr);
                } else {
                    sh_o = Some(arr);
                }
            }
            other => {
                return Err(UsdError::UnsupportedUsdcFeature(format!(
                    "attribute typeName '{other}' not supported by in-process reader"
                )));
            }
        }
    }

    let positions = points.ok_or_else(|| UsdError::MalformedUsdc("missing 'points'".into()))?;
    let orientations =
        orient.ok_or_else(|| UsdError::MalformedUsdc("missing 'orientations'".into()))?;
    let scales = scales_o.ok_or_else(|| UsdError::MalformedUsdc("missing 'scales'".into()))?;
    let opacities = opac_o.ok_or_else(|| UsdError::MalformedUsdc("missing 'opacities'".into()))?;
    let colors = colors_o.ok_or_else(|| UsdError::MalformedUsdc("missing 'colorsDC'".into()))?;

    let n = positions.len();
    if [
        orientations.len(),
        scales.len(),
        opacities.len(),
        colors.len(),
    ]
    .iter()
    .any(|&l| l != n)
    {
        return Err(UsdError::MalformedUsdc("attribute length mismatch".into()));
    }

    let sh_per = sh_o.as_ref().map(|v| v.len() / n);
    let mut splats = Vec::with_capacity(n);
    for i in 0..n {
        // `read_array_quatf` returns (w, x, y, z); IR is (x, y, z, w).
        let q = orientations[i];
        let rotation = [q[1], q[2], q[3], q[0]];
        let color = if let (Some(coeffs), Some(stride)) = (&sh_o, sh_per) {
            let off = i * stride;
            if off + stride <= coeffs.len() && coeffs[off..off + stride].iter().any(|&c| c != 0.0) {
                let degree = match stride {
                    3 => 0,
                    12 => 1,
                    27 => 2,
                    48 => 3,
                    _ => 3,
                };
                Color::Sh {
                    degree,
                    coeffs: coeffs[off..off + stride].to_vec(),
                }
            } else {
                Color::Rgb(colors[i])
            }
        } else {
            Color::Rgb(colors[i])
        };
        splats.push(Splat {
            position: positions[i],
            rotation,
            scale: scales[i],
            opacity: opacities[i],
            color,
        });
    }
    let mut scene = SplatScene::new();
    scene.splats = splats;
    Ok(scene)
}

fn read_tokens(section: &[u8]) -> Result<Vec<String>, UsdError> {
    // Version 0.0.1 layout (uncompressed):
    //   u64 numTokens
    //   u64 totalBytes
    //   <totalBytes> null-terminated UTF-8 strings.
    if section.len() < 16 {
        return Err(UsdError::MalformedUsdc("tokens section short".into()));
    }
    let n = u64::from_le_bytes(section[..8].try_into().unwrap()) as usize;
    let total_bytes = u64::from_le_bytes(section[8..16].try_into().unwrap()) as usize;
    if 16 + total_bytes > section.len() {
        return Err(UsdError::MalformedUsdc("tokens payload truncated".into()));
    }
    let raw = &section[16..16 + total_bytes];
    let mut toks = Vec::with_capacity(n);
    let mut start = 0;
    for i in 0..raw.len() {
        if raw[i] == 0 {
            toks.push(
                std::str::from_utf8(&raw[start..i])
                    .map_err(|e| UsdError::MalformedUsdc(format!("token: {e}")))?
                    .to_string(),
            );
            start = i + 1;
            if toks.len() == n {
                break;
            }
        }
    }
    if toks.len() != n {
        return Err(UsdError::MalformedUsdc(format!(
            "tokens: parsed {} but section header claimed {}",
            toks.len(),
            n
        )));
    }
    Ok(toks)
}

/// Read an array of `n_elements * stride` little-endian f32 values, where
/// the on-disk header is `[u32 shape=1, u32 numElements, T data...]`. We pass
/// the element stride (1 for `float[]`, 3 for `Vec3f`, 4 for `Quatf`) so the
/// reader can compute the byte size correctly — Pixar's layout uses
/// `numElements`, not `numFloats`.
fn read_array_f32_stride(bytes: &[u8], off: usize, stride: usize) -> Result<Vec<f32>, UsdError> {
    if off + 8 > bytes.len() {
        return Err(UsdError::MalformedUsdc("array header oob".into()));
    }
    let shape = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    let n_elements = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
    if shape != 1 {
        return Err(UsdError::UnsupportedUsdcFeature(format!(
            "array shape {shape} (expected 1)"
        )));
    }
    let n_floats = n_elements * stride;
    let data_off = off + 8;
    if data_off + n_floats * 4 > bytes.len() {
        return Err(UsdError::MalformedUsdc("array data oob".into()));
    }
    let mut out = Vec::with_capacity(n_floats);
    for i in 0..n_floats {
        let pp = data_off + i * 4;
        out.push(f32::from_le_bytes(bytes[pp..pp + 4].try_into().unwrap()));
    }
    Ok(out)
}

fn read_array_f32(bytes: &[u8], off: usize) -> Result<Vec<f32>, UsdError> {
    read_array_f32_stride(bytes, off, 1)
}

fn read_array_vec3f(bytes: &[u8], off: usize) -> Result<Vec<[f32; 3]>, UsdError> {
    let raw = read_array_f32_stride(bytes, off, 3)?;
    if raw.len() % 3 != 0 {
        return Err(UsdError::MalformedUsdc(
            "vec3 array length not multiple of 3".into(),
        ));
    }
    Ok(raw.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
}

fn read_array_quatf(bytes: &[u8], off: usize) -> Result<Vec<[f32; 4]>, UsdError> {
    // Wire layout is (x, y, z, w). Return (w, x, y, z) tuples to match the
    // writer's pre-shuffle convention.
    let raw = read_array_f32_stride(bytes, off, 4)?;
    if raw.len() % 4 != 0 {
        return Err(UsdError::MalformedUsdc(
            "quat array length not multiple of 4".into(),
        ));
    }
    Ok(raw.chunks(4).map(|c| [c[3], c[0], c[1], c[2]]).collect())
}

#[allow(dead_code)]
fn vr_type_enum(rep: u64) -> Option<TypeEnum> {
    match vr_type(rep) {
        1 => Some(TypeEnum::Bool),
        8 => Some(TypeEnum::Float),
        9 => Some(TypeEnum::Double),
        11 => Some(TypeEnum::Token),
        17 => Some(TypeEnum::Quatf),
        24 => Some(TypeEnum::Vec3f),
        41 => Some(TypeEnum::TokenVector),
        42 => Some(TypeEnum::Specifier),
        43 => Some(TypeEnum::Permission),
        44 => Some(TypeEnum::Variability),
        _ => None,
    }
}
