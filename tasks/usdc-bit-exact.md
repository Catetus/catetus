# USDC bit-exact-as-USDA against `usdcat`

## Target
Version 0.0.1 of Pixar Crate format (simplest; uncompressed Fields/FieldSets/Paths/Specs; LZ4-only-on-Tokens).
`usdcat` (Apple USD Tools 0.25.2 installed at `/usr/bin/usdcat`) reads all minor versions <= software version, so 0.0.1 is forward-compatible.

## Status
- [x] Reverse-engineered Pixar source (pxr/usd/sdf/crateFile.{h,cpp}, crateDataTypes.h, fastCompression.cpp).
- [x] Decoded a real usdcat-emitted USDC to confirm format details.
- [ ] Implement `write_usdc` (version 0.0.1, deterministic).
- [ ] Implement `read_usdc` (version 0.0.1 only, with version-gate).
- [ ] Wire into CLI convert: `catetus convert in.usda out.usdc --to usdc`.
- [ ] `scripts/usdc-roundtrip.sh` against 3 reference USDA files.
- [ ] `tests/usdc_roundtrip.rs` (in-process: write USDC -> read USDC -> compare scene).
- [ ] `tests/usdc_usdcat_validation.rs` behind `usdcat-validation` feature flag.
- [ ] `docs/openusd-conformance.md`.
- [ ] `crates/catetus-usd/SPEC-GAPS.md`.

## Format details (locked)

Bootstrap (88 bytes at offset 0):
- 8 bytes "PXR-USDC"
- 8 bytes version[8]: major, minor, patch, 5 zero
- int64 tocOffset
- 64 bytes reserved (zero)

Sections (each named, written sequentially, named in TOC at end):
- TOKENS: u64 numTokens, u64 uncompressedSize, u64 compressedSize, then `compressedSize` bytes of TfFastCompression payload (single chunk: 1 byte 0x00 then LZ4 block). The decompressed stream is `numTokens` C strings (null-terminated). Token 0 must be ";-)" (sentinel; sidesteps PathTree negative-index issue).
- STRINGS: u64 numStrings, then numStrings * u32 (each is a token index).
- FIELDS (version 0.0.1): u64 numFields, then numFields * 16-byte Field { u32 _padding, u32 tokenIndex, u64 valueRep }.
- FIELDSETS (version 0.0.1): u64 numEntries, then numEntries * u32 (each is a field index; sentinel ~0u terminates each set).
- PATHS (version 0.0.1): u64 numPaths, then path-tree of `_PathItemHeader_0_0_1` records: 16 bytes each { u32 _padding, u32 pathIndex, u32 elementTokenIndex, u8 bits, 3 bytes trailing pad }. If both child and sibling, an i64 sibling-offset follows the header.
- SPECS (version 0.0.1): u64 numSpecs, then numSpecs * 16-byte Spec_0_0_1 { u32 _padding, u32 pathIndex, u32 fieldSetIndex, u32 specType } (specType actually 4 bytes from enum).

TOC (at tocOffset): u64 numSections, then numSections * { 16-byte name (zero-pad), i64 start, i64 size }.

ValueRep is u64:
- bits 63: IsArray
- bit 62: IsInlined  (for "always inlined" types: Token, String, SdfPath, SdfAssetPath, and POD <= 4 bytes)
- bit 61: IsCompressed
- bit 60: IsArrayEdit
- bits 56..48: TypeEnum (8 bits, 0-60)
- bits 47..0: payload

For an inlined Token: payload = tokenIndex (u32). For an array<float>: type=Float(8), IsArray=true, payload=file offset of `[u32 shape=1][u32 size][float data...]`.

## Schema mapping
Single `Xform "World"` containing one `ParticleField3DGaussianSplat "Splats"` prim with:
- typeName="ParticleField3DGaussianSplat"
- attribute `points` (point3f[]) = Vec3f array
- attribute `orientations` (quatf[]) = Quatf array (w,x,y,z)
- attribute `scales` (float3[]) = Vec3f array
- attribute `opacities` (float[]) = Float array
- attribute `colorsDC` (color3f[]) = Vec3f array
- optional `catetus:shCoefficients` custom float[] when scene has SH

Each attribute spec needs: typeName (Token), default (the array), custom (bool, only if true), variability (Variability=Varying).

## Spec gap risks
- 26.03's ParticleField3DGaussianSplat spec doesn't yet fully nail SH coefficient packing; we emit a custom-namespaced attribute. Document in SPEC-GAPS.md.
