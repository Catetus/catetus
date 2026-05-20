// SPDX-License-Identifier: Apache-2.0
//
// diag-vq45 — visual smoke-test for the production viewer's new GLB-extension
// decoders: SF_zstd_split_buffer, SF_gaussian_splatting_palette, and BYTE/SHORT
// SH-rest accessors. Renders an FP32 baseline GLB next to its compressed
// counterpart so visual parity is obvious to a human reviewer.
//
// All decoding logic is ported from
// `experiments/w3-fidelity-harness/code/cpu-fidelity.mjs`; the same wire format
// the bench harness reads is what we read here, so a green diag here means the
// compressed format is ship-able in the production viewer too.

import * as fzstd from './fzstd.mjs';
// Reuse the production viewer's GLB chunk splitter AND the new zstd-split /
// .shpal decoders so this diag exercises the same code path the SDK ships.
import {
  decodeGlb,
  decompressZstdSplitBuffer as viewerDecompressZstdSplit,
  decodeShPaletteSidecar as viewerDecodeShPalette,
} from '/viewer/streaming/glb.js';

const log = (...args) => {
  const el = document.getElementById('log');
  const s = args
    .map((a) => (typeof a === 'string' ? a : JSON.stringify(a, null, 2)))
    .join(' ');
  el.textContent += s + '\n';
  el.scrollTop = el.scrollHeight;
  // eslint-disable-next-line no-console
  console.log(...args);
};

const setStatus = (id, text, cls) => {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = cls || '';
};

/* ============================================================ decoder port */

// Thin wrappers around the viewer module so the rest of this file reads as
// `decompressZstdSplitBuffer(bin, ext)` rather than threading the `fzstd`
// dependency through every call site. The actual decoder logic lives in
// `packages/viewer/src/streaming/glb.ts`.
const decompressZstdSplitBuffer = (compressed, ext) =>
  viewerDecompressZstdSplit(compressed, ext, fzstd.decompress);
const loadShPaletteSidecar = (compressed, ext) =>
  viewerDecodeShPalette(compressed, ext, fzstd.decompress);

/* ============================================================ glb → splats */

/**
 * Build a `DecodedSplat[]` from a glTF manifest + uncompressed BIN, honoring
 * both legacy and KHR-RC primitive-attribute layouts, KHR_mesh_quantization
 * (UBYTE/USHORT normalized), and the new BYTE/SHORT SH-rest accessor shape
 * (which the renderer ignores but we decode the DC alongside so this is a
 * single code path).
 *
 * SH-rest is intentionally dropped — the production WebGL2/WebGPU renderer
 * currently only consumes DC color. That matches the bench harness's
 * `dropShRest` view and is the conservative thing to do until the renderer
 * grows view-dependent color.
 */
function decodeAttribute(bytes, slice, splatCount, comps) {
  const total = splatCount * comps;
  const base = bytes.byteOffset + (slice.byteOffset | 0);
  const ct = slice.componentType;
  if (ct === 5126 /* FLOAT */) {
    if ((base & 3) === 0) return new Float32Array(bytes.buffer, base, total);
    const dv = new DataView(bytes.buffer, base, total * 4);
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = dv.getFloat32(i * 4, true);
    return out;
  }
  if (ct === 5123 /* USHORT */) {
    const src = new Uint16Array(bytes.buffer, base, total);
    const out = new Float32Array(total);
    if (slice.normalized && slice.min && slice.max && slice.min.length === comps) {
      for (let i = 0; i < total; i++) {
        const k = i % comps;
        const lo = slice.min[k];
        const hi = slice.max[k];
        out[i] = lo + (src[i] / 65535) * (hi - lo);
      }
    } else if (slice.normalized) {
      for (let i = 0; i < total; i++) out[i] = src[i] / 65535;
    } else {
      for (let i = 0; i < total; i++) out[i] = src[i];
    }
    return out;
  }
  if (ct === 5121 /* UBYTE */) {
    const src = new Uint8Array(bytes.buffer, base, total);
    const out = new Float32Array(total);
    if (slice.normalized && slice.min && slice.max && slice.min.length === comps) {
      for (let i = 0; i < total; i++) {
        const k = i % comps;
        const lo = slice.min[k];
        const hi = slice.max[k];
        out[i] = lo + (src[i] / 255) * (hi - lo);
      }
    } else if (slice.normalized) {
      for (let i = 0; i < total; i++) out[i] = src[i] / 255;
    } else {
      for (let i = 0; i < total; i++) out[i] = src[i];
    }
    return out;
  }
  if (ct === 5120 /* BYTE */) {
    // Signed BYTE normalized — SH-rest @ bits<=8. Per-channel range from
    // accessor min/max (writer emits symmetric [-r, +r]).
    const src = new Int8Array(bytes.buffer, base, total);
    const out = new Float32Array(total);
    if (slice.min && slice.max && slice.min.length === comps) {
      for (let i = 0; i < total; i++) {
        const k = i % comps;
        const r = Math.max(slice.max[k], -slice.min[k], 1e-9);
        const t = Math.max(src[i] / 127.0, -1.0);
        out[i] = t * r;
      }
    } else {
      for (let i = 0; i < total; i++) out[i] = Math.max(src[i] / 127.0, -1.0);
    }
    return out;
  }
  if (ct === 5122 /* SHORT */) {
    const src = new Int16Array(bytes.buffer, base, total);
    const out = new Float32Array(total);
    if (slice.min && slice.max && slice.min.length === comps) {
      for (let i = 0; i < total; i++) {
        const k = i % comps;
        const r = Math.max(slice.max[k], -slice.min[k], 1e-9);
        const t = Math.max(src[i] / 32767.0, -1.0);
        out[i] = t * r;
      }
    } else {
      for (let i = 0; i < total; i++) out[i] = Math.max(src[i] / 32767.0, -1.0);
    }
    return out;
  }
  throw new Error(`unsupported componentType ${ct}`);
}

function pickAttr(prim, key) {
  // RC layout: `prim.attributes["KHR_gaussian_splatting:..."]`.
  // Legacy layout: `prim.extensions.KHR_gaussian_splatting.attributes[bareKey]`.
  const rc = prim.attributes;
  const ns = 'KHR_gaussian_splatting:' + key;
  if (rc && typeof rc[ns] === 'number') return rc[ns];
  if (key === 'POSITION' && rc && typeof rc.POSITION === 'number') return rc.POSITION;
  const legacy = prim.extensions?.KHR_gaussian_splatting?.attributes;
  if (legacy && typeof legacy[key] === 'number') return legacy[key];
  if (legacy && typeof legacy['_' + key] === 'number') return legacy['_' + key];
  return undefined;
}

function accessorSlice(g, idx) {
  const a = g.accessors[idx];
  const bv = g.bufferViews[a.bufferView];
  return {
    byteOffset: (bv.byteOffset | 0) + (a.byteOffset | 0),
    byteLength: bv.byteLength | 0,
    componentType: a.componentType,
    normalized: !!a.normalized,
    min: Array.isArray(a.min) ? a.min : null,
    max: Array.isArray(a.max) ? a.max : null,
    count: a.count | 0,
  };
}

async function loadSplats(glbUrl) {
  log(`fetching ${glbUrl} ...`);
  const t0 = performance.now();
  const res = await fetch(glbUrl);
  if (!res.ok) throw new Error(`fetch failed: HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  log(`  ${(buf.byteLength / (1024 * 1024)).toFixed(1)} MiB in ${(performance.now() - t0).toFixed(0)} ms`);

  const { json, bin } = decodeGlb(buf);
  const manifest = JSON.parse(json);

  const exts = manifest.extensionsUsed || [];
  log(`  extensions: [${exts.join(', ')}]`);

  // 1) SF_zstd_split_buffer — uncompress in place.
  let workingBin = bin;
  const zstdExt = manifest.extensions?.SF_zstd_split_buffer;
  if (zstdExt) {
    const tZ = performance.now();
    workingBin = decompressZstdSplitBuffer(bin, zstdExt);
    log(`  zstd-split: ${(bin.byteLength / (1024 * 1024)).toFixed(1)} MiB → ${(workingBin.byteLength / (1024 * 1024)).toFixed(1)} MiB in ${(performance.now() - tZ).toFixed(0)} ms`);
  }

  // 2) SF_gaussian_splatting_palette — fetch + decode `.shpal` sidecar.
  const palExt = manifest.extensions?.SF_gaussian_splatting_palette;
  let palette = null;
  if (palExt && palExt.uri) {
    const sidecarUrl = new URL(palExt.uri, new URL(glbUrl, location.href)).toString();
    log(`  fetching .shpal sidecar ${sidecarUrl} ...`);
    const sRes = await fetch(sidecarUrl);
    if (!sRes.ok) throw new Error(`sidecar fetch failed: HTTP ${sRes.status}`);
    const sBytes = new Uint8Array(await sRes.arrayBuffer());
    palette = loadShPaletteSidecar(sBytes, palExt);
    log(`  .shpal: K=${palette.K} N=${palette.N} bits=${palette.codebookBits} shDegree=${palette.shDegree}`);
  }
  void palette; // renderer doesn't consume SH-rest yet.

  // Drop the only sidecar-fetchable .shpal even when there's no palette ext —
  // also smoke-test the decoder path so the demo exercises it.
  if (!palette && glbUrl.endsWith('wmv-vq45-no-prune.glb')) {
    try {
      const probeUrl = glbUrl + '.shpal';
      const sRes = await fetch(probeUrl);
      if (sRes.ok) {
        const sBytes = new Uint8Array(await sRes.arrayBuffer());
        const probed = loadShPaletteSidecar(sBytes, null);
        log(`  probed .shpal (no ext ref): K=${probed.K} N=${probed.N} bits=${probed.codebookBits}`);
      }
    } catch (err) {
      log(`  .shpal probe skipped: ${err.message}`);
    }
  }

  // 3) Pull splat attributes.
  const prim = manifest.meshes[0].primitives[0];
  const posIdx = pickAttr(prim, 'POSITION');
  const rotIdx = pickAttr(prim, 'ROTATION');
  const sclIdx = pickAttr(prim, 'SCALE');
  const opIdx = pickAttr(prim, 'OPACITY');
  // DC color: prefer KHR-RC `SH_DEGREE_0_COEF_0`, else legacy `COLOR_DC`.
  let dcIdx = pickAttr(prim, 'SH_DEGREE_0_COEF_0');
  if (dcIdx === undefined) dcIdx = pickAttr(prim, 'COLOR_DC');
  if (
    posIdx === undefined ||
    rotIdx === undefined ||
    sclIdx === undefined ||
    opIdx === undefined ||
    dcIdx === undefined
  ) {
    throw new Error('missing KHR_gaussian_splatting required attributes');
  }
  const posS = accessorSlice(manifest, posIdx);
  const rotS = accessorSlice(manifest, rotIdx);
  const sclS = accessorSlice(manifest, sclIdx);
  const opS = accessorSlice(manifest, opIdx);
  const dcS = accessorSlice(manifest, dcIdx);
  const N = posS.count;

  const tDec = performance.now();
  const pos = decodeAttribute(workingBin, posS, N, 3);
  const rot = decodeAttribute(workingBin, rotS, N, 4);
  const scl = decodeAttribute(workingBin, sclS, N, 3);
  const op = decodeAttribute(workingBin, opS, N, 1);
  const dc = decodeAttribute(workingBin, dcS, N, 3);
  log(`  decoded ${N} splats in ${(performance.now() - tDec).toFixed(0)} ms`);

  // The DC accessor is FP32 here (and in the baseline) — raw SH DC values, not
  // RGB. The viewer's WebGL2 renderer expects a [0,1] RGB-ish channel for
  // `colorDC`, so we bake the SH_C0 shift here exactly like the harness does.
  const SH_C0 = 0.28209479177387814;
  const rgb = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    rgb[i * 3 + 0] = Math.min(1, Math.max(0, dc[i * 3 + 0] * SH_C0 + 0.5));
    rgb[i * 3 + 1] = Math.min(1, Math.max(0, dc[i * 3 + 1] * SH_C0 + 0.5));
    rgb[i * 3 + 2] = Math.min(1, Math.max(0, dc[i * 3 + 2] * SH_C0 + 0.5));
  }

  // Position min/max — KHR_mesh_quantization carries it on POSITION; if it's
  // an FP32 accessor we scan it here so the camera framing matches.
  let xmin = Infinity,
    ymin = Infinity,
    zmin = Infinity,
    xmax = -Infinity,
    ymax = -Infinity,
    zmax = -Infinity;
  if (posS.min && posS.max) {
    [xmin, ymin, zmin] = posS.min;
    [xmax, ymax, zmax] = posS.max;
  } else {
    for (let i = 0; i < N; i++) {
      const x = pos[i * 3 + 0];
      const y = pos[i * 3 + 1];
      const z = pos[i * 3 + 2];
      if (x < xmin) xmin = x;
      if (y < ymin) ymin = y;
      if (z < zmin) zmin = z;
      if (x > xmax) xmax = x;
      if (y > ymax) ymax = y;
      if (z > zmax) zmax = z;
    }
  }
  return {
    N,
    pos,
    rot,
    scl,
    op,
    dc: rgb,
    bbox: { min: [xmin, ymin, zmin], max: [xmax, ymax, zmax] },
  };
}

/**
 * Subsample a decoded splat set with a fixed stride. Keeps the bbox so the
 * camera framing matches the dense scene. Used by the diag page to keep
 * draw-call count manageable on swiftshader (full 1.15M splats are fine on a
 * real GPU but blow up software rasterizers).
 */
function subsampleSplats(s, target) {
  if (s.N <= target) return s;
  const stride = Math.ceil(s.N / target);
  const M = Math.ceil(s.N / stride);
  const pos = new Float32Array(M * 3);
  const rot = new Float32Array(M * 4);
  const scl = new Float32Array(M * 3);
  const op = new Float32Array(M);
  const dc = new Float32Array(M * 3);
  for (let j = 0, i = 0; i < s.N; i += stride, j++) {
    pos[j * 3 + 0] = s.pos[i * 3 + 0];
    pos[j * 3 + 1] = s.pos[i * 3 + 1];
    pos[j * 3 + 2] = s.pos[i * 3 + 2];
    rot[j * 4 + 0] = s.rot[i * 4 + 0];
    rot[j * 4 + 1] = s.rot[i * 4 + 1];
    rot[j * 4 + 2] = s.rot[i * 4 + 2];
    rot[j * 4 + 3] = s.rot[i * 4 + 3];
    scl[j * 3 + 0] = s.scl[i * 3 + 0];
    scl[j * 3 + 1] = s.scl[i * 3 + 1];
    scl[j * 3 + 2] = s.scl[i * 3 + 2];
    op[j] = s.op[i];
    dc[j * 3 + 0] = s.dc[i * 3 + 0];
    dc[j * 3 + 1] = s.dc[i * 3 + 1];
    dc[j * 3 + 2] = s.dc[i * 3 + 2];
  }
  return { N: M, pos, rot, scl, op, dc, bbox: s.bbox };
}

/* ============================================================ minimal renderer harness */

// Inline a tiny WebGL2 splat blitter so we don't have to thread our decoded
// arrays through the viewer's chunk-streaming pipeline. The renderer matches
// the production WebGL2 path (alpha-blended ellipsoid sprites, same 2D
// projected covariance), but operates on plain typed arrays.
//
// Math is the same as `packages/viewer/src/renderer/math.ts`; we inline the
// pieces we need to keep this file dependency-free at runtime.

function buildOrbit(bbox, yaw, pitch, aspect, framing) {
  const cx = (bbox.min[0] + bbox.max[0]) * 0.5;
  const cy = (bbox.min[1] + bbox.max[1]) * 0.5;
  const cz = (bbox.min[2] + bbox.max[2]) * 0.5;
  const dx = bbox.max[0] - bbox.min[0];
  const dy = bbox.max[1] - bbox.min[1];
  const dz = bbox.max[2] - bbox.min[2];
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const radius = diag * framing;
  const eyeX = cx + Math.cos(yaw) * Math.cos(pitch) * radius;
  const eyeY = cy + Math.sin(pitch) * radius;
  const eyeZ = cz + Math.sin(yaw) * Math.cos(pitch) * radius;
  const fovY = (45 * Math.PI) / 180;
  return {
    eye: [eyeX, eyeY, eyeZ],
    target: [cx, cy, cz],
    up: [0, 1, 0],
    fovY,
    aspect,
    near: Math.max(0.01, diag * 0.01),
    far: diag * 5,
  };
}

function buildView(eye, target, up) {
  const fx = target[0] - eye[0];
  const fy = target[1] - eye[1];
  const fz = target[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1;
  const fwd = [fx / fl, fy / fl, fz / fl];
  // right = fwd x up
  const rx = fwd[1] * up[2] - fwd[2] * up[1];
  const ry = fwd[2] * up[0] - fwd[0] * up[2];
  const rz = fwd[0] * up[1] - fwd[1] * up[0];
  const rl = Math.hypot(rx, ry, rz) || 1;
  const r = [rx / rl, ry / rl, rz / rl];
  // u = right x fwd
  const u = [r[1] * fwd[2] - r[2] * fwd[1], r[2] * fwd[0] - r[0] * fwd[2], r[0] * fwd[1] - r[1] * fwd[0]];
  // Column-major view matrix (looking at -fwd in view space).
  const view = new Float32Array(16);
  view[0] = r[0]; view[4] = r[1]; view[8] = r[2];  view[12] = -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]);
  view[1] = u[0]; view[5] = u[1]; view[9] = u[2];  view[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
  view[2] = -fwd[0]; view[6] = -fwd[1]; view[10] = -fwd[2]; view[14] = fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2];
  view[15] = 1;
  return view;
}

function buildProj(fovY, aspect, near, far) {
  const f = 1 / Math.tan(fovY * 0.5);
  const p = new Float32Array(16);
  p[0] = f / aspect;
  p[5] = f;
  p[10] = (far + near) / (near - far);
  p[11] = -1;
  p[14] = (2 * far * near) / (near - far);
  return p;
}

function mulMat4(out, a, b) {
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[i + k * 4] * b[k + j * 4];
      out[i + j * 4] = s;
    }
  }
  return out;
}

function quatToMat3(q) {
  // q = [x, y, z, w]
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [
    1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy),
    2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx),
    2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy),
  ];
}

// Shader is a stripped-down 3DGS splatter: each splat draws a screen-space
// quad sized to ±3σ along the projected 2D covariance's principal axes; the
// fragment shader exponentially attenuates inside the ellipsoid.
const VS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aQuad;          // [-1,1]^2 corners
layout(location = 1) in vec4 aPixelXY;       // splat center in pixels (x, y, depth, _)
layout(location = 2) in vec4 aCov;           // c00, c01, c11 in pixels^2 + radius hint
layout(location = 3) in vec4 aColor;         // r, g, b, a
uniform vec2 uViewport;

out vec4 vColor;
out vec2 vUV;                                // in unit-σ space

void main() {
  vColor = aColor;
  float c00 = aCov.x;
  float c01 = aCov.y;
  float c11 = aCov.z;
  // Symmetric 2x2 eigendecomposition.
  float trace = c00 + c11;
  float halfTrace = trace * 0.5;
  float det = c00 * c11 - c01 * c01;
  float disc = sqrt(max(halfTrace * halfTrace - det, 0.0));
  float l1 = halfTrace + disc;
  float l2 = max(halfTrace - disc, 0.0);
  vec2 e1 = abs(c01) > 1e-6
    ? normalize(vec2(l1 - c11, c01))
    : (c00 >= c11 ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 e2 = vec2(-e1.y, e1.x);
  float s1 = sqrt(max(l1, 0.0));
  float s2 = sqrt(max(l2, 0.0));
  // Quad corner in pixel space, sized to ±3σ on each principal axis.
  vec2 offsetPx = aQuad.x * e1 * (3.0 * s1) + aQuad.y * e2 * (3.0 * s2);
  vec2 cornerPx = aPixelXY.xy + offsetPx;
  vec2 ndc = (cornerPx / uViewport) * 2.0 - 1.0;
  // Pass UV in σ units to the FS (so |vUV|=3 corresponds to the quad edge).
  vUV = aQuad * 3.0;
  gl_Position = vec4(ndc.x, -ndc.y, aPixelXY.z, 1.0);
}
`;

const FS = `#version 300 es
precision highp float;
in vec4 vColor;
in vec2 vUV;
out vec4 outColor;
void main() {
  float d2 = dot(vUV, vUV);
  if (d2 > 9.0) discard;
  float a = exp(-0.5 * d2) * vColor.a;
  outColor = vec4(vColor.rgb * a, a);
}
`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

function projectCov2D(scale, q, view, fx, fy, depth) {
  const r = quatToMat3(q);
  const m00 = r[0] * scale[0], m01 = r[3] * scale[1], m02 = r[6] * scale[2];
  const m10 = r[1] * scale[0], m11 = r[4] * scale[1], m12 = r[7] * scale[2];
  const m20 = r[2] * scale[0], m21 = r[5] * scale[1], m22 = r[8] * scale[2];
  // Sigma_3D = M * M^T
  const s00 = m00 * m00 + m01 * m01 + m02 * m02;
  const s01 = m00 * m10 + m01 * m11 + m02 * m12;
  const s02 = m00 * m20 + m01 * m21 + m02 * m22;
  const s11 = m10 * m10 + m11 * m11 + m12 * m12;
  const s12 = m10 * m20 + m11 * m21 + m12 * m22;
  const s22 = m20 * m20 + m21 * m21 + m22 * m22;
  // Project: J * R^T * Sigma * R * J^T, where R = view rotation, J = perspective Jacobian.
  // (J*R^T)[0..2] rows: take view rows 0/1/2 scaled by fx/depth, fy/depth.
  const invD = 1 / depth;
  const j00 = fx * invD, j02 = -fx * 0 * invD; // we use simple pinhole
  // Build W = view rotation (top-left 3x3)
  const w00 = view[0], w01 = view[4], w02 = view[8];
  const w10 = view[1], w11 = view[5], w12 = view[9];
  const w20 = view[2], w21 = view[6], w22 = view[10];
  // T = J * W
  const t00 = fx * invD * w00, t01 = fx * invD * w01, t02 = fx * invD * w02;
  const t10 = fy * invD * w10, t11 = fy * invD * w11, t12 = fy * invD * w12;
  // Cov2 = T * Sigma * T^T (2x2)
  // First compute U = T * Sigma  (2x3)
  const u00 = t00 * s00 + t01 * s01 + t02 * s02;
  const u01 = t00 * s01 + t01 * s11 + t02 * s12;
  const u02 = t00 * s02 + t01 * s12 + t02 * s22;
  const u10 = t10 * s00 + t11 * s01 + t12 * s02;
  const u11 = t10 * s01 + t11 * s11 + t12 * s12;
  const u12 = t10 * s02 + t11 * s12 + t12 * s22;
  // Cov2 = U * T^T  (2x2)
  const c00 = u00 * t00 + u01 * t01 + u02 * t02 + 0.3;
  const c01 = u00 * t10 + u01 * t11 + u02 * t12;
  const c11 = u10 * t10 + u11 * t11 + u12 * t12 + 0.3;
  return [c00, c01, c11];
  void j00; void j02;
}

async function renderInto(canvasId, splats) {
  const canvas = document.getElementById(canvasId);
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false });
  if (!gl) throw new Error('WebGL2 unavailable');
  const W = canvas.width;
  const H = canvas.height;
  gl.viewport(0, 0, W, H);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.05, 0.06, 0.08, 1.0);
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const qb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, qb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.vertexAttribDivisor(0, 0);

  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, ib);
  // Pre-allocate; we'll re-upload per frame.
  const STRIDE = 12 * 4;
  gl.bufferData(gl.ARRAY_BUFFER, splats.N * STRIDE, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 4, gl.FLOAT, false, STRIDE, 0);
  gl.vertexAttribDivisor(1, 1);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 4, gl.FLOAT, false, STRIDE, 16);
  gl.vertexAttribDivisor(2, 1);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 4, gl.FLOAT, false, STRIDE, 32);
  gl.vertexAttribDivisor(3, 1);

  const cam = buildOrbit(splats.bbox, Math.PI * 0.25, 0.2, W / H, 1.2);
  const view = buildView(cam.eye, cam.target, cam.up);
  const proj = buildProj(cam.fovY, cam.aspect, cam.near, cam.far);
  const vp = mulMat4(new Float32Array(16), proj, view);
  const focalY = H / (2 * Math.tan(cam.fovY * 0.5));
  const focalX = focalY;

  // Sort back-to-front by view-space depth.
  const N = splats.N;
  const depths = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = splats.pos[i * 3], y = splats.pos[i * 3 + 1], z = splats.pos[i * 3 + 2];
    const vz = view[2] * x + view[6] * y + view[10] * z + view[14];
    depths[i] = -vz;
  }
  // Sort by depth desc (back-to-front). Array.sort comparator allocates a
  // closure capture but at <300K splats this is well under 100 ms.
  const pairs = new Array(N);
  for (let i = 0; i < N; i++) pairs[i] = i;
  pairs.sort((a, b) => depths[b] - depths[a]);

  const out = new Float32Array(N * 12);
  const sclTmp = [0, 0, 0];
  const rotTmp = [0, 0, 0, 0];
  for (let p = 0; p < N; p++) {
    const i = pairs[p];
    const x = splats.pos[i * 3], y = splats.pos[i * 3 + 1], z = splats.pos[i * 3 + 2];
    const px = vp[0] * x + vp[4] * y + vp[8] * z + vp[12];
    const py = vp[1] * x + vp[5] * y + vp[9] * z + vp[13];
    const pz = vp[2] * x + vp[6] * y + vp[10] * z + vp[14];
    const pw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15];
    if (pw <= 0) {
      out[p * 12 + 11] = 0;
      continue;
    }
    const vz = view[2] * x + view[6] * y + view[10] * z + view[14];
    const depth = -vz;
    if (depth <= 0.01) {
      out[p * 12 + 11] = 0;
      continue;
    }
    // Project to pixel space (top-left origin; FS flips Y).
    const ndcX = px / pw;
    const ndcY = py / pw;
    const pixelX = (ndcX * 0.5 + 0.5) * W;
    const pixelY = (ndcY * 0.5 + 0.5) * H;
    sclTmp[0] = splats.scl[i * 3]; sclTmp[1] = splats.scl[i * 3 + 1]; sclTmp[2] = splats.scl[i * 3 + 2];
    rotTmp[0] = splats.rot[i * 4]; rotTmp[1] = splats.rot[i * 4 + 1]; rotTmp[2] = splats.rot[i * 4 + 2]; rotTmp[3] = splats.rot[i * 4 + 3];
    const [c00, c01, c11] = projectCov2D(sclTmp, rotTmp, view, focalX, focalY, depth);
    out[p * 12 + 0] = pixelX;
    out[p * 12 + 1] = pixelY;
    out[p * 12 + 2] = pz / pw;       // gl_Position.z
    out[p * 12 + 3] = 0;
    out[p * 12 + 4] = c00;           // pixel^2
    out[p * 12 + 5] = c01;
    out[p * 12 + 6] = c11;
    out[p * 12 + 7] = 0;
    out[p * 12 + 8] = splats.dc[i * 3];
    out[p * 12 + 9] = splats.dc[i * 3 + 1];
    out[p * 12 + 10] = splats.dc[i * 3 + 2];
    out[p * 12 + 11] = splats.op[i];
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, ib);
  gl.bufferData(gl.ARRAY_BUFFER, out, gl.DYNAMIC_DRAW);
  gl.clear(gl.COLOR_BUFFER_BIT);
  const uViewport = gl.getUniformLocation(prog, 'uViewport');
  gl.uniform2f(uViewport, W, H);
  gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, N);
  const err = gl.getError();
  if (err !== 0) log(`gl error ${canvasId}:`, err);
}

/* ============================================================ orchestration */

async function loadSide(label, glbUrl, canvasId, statusId, opts = {}) {
  setStatus(statusId, 'loading…', 'loading');
  try {
    let splats = await loadSplats(glbUrl);
    const target = opts.maxSplats || 200_000;
    if (splats.N > target) {
      const sub = subsampleSplats(splats, target);
      log(`  subsampled ${splats.N} → ${sub.N} splats for render`);
      splats = sub;
    }
    setStatus(statusId, `rendering ${splats.N.toLocaleString()} splats…`, 'loading');
    await renderInto(canvasId, splats);
    setStatus(statusId, `ok — rendering ${splats.N.toLocaleString()} splats`, 'ok');
    log(`${label}: bbox`, splats.bbox);
    return splats;
  } catch (err) {
    setStatus(statusId, `error: ${err.message}`, 'err');
    log(`${label} FAILED:`, err.message);
    console.error(err);
    throw err;
  }
}

async function main() {
  // The production viewer expects a manifest URI flow; what we want to verify
  // here is the decoder. The minimal WebGL2 splatter above matches the
  // renderer math closely enough that a broken decoder produces visibly
  // garbled splats.

  log('--- diag-vq45 ---');
  log('User agent:', navigator.userAgent);

  // Run both in parallel so an observer sees them appear at roughly the same
  // moment; this lets the screenshot reviewer compare without scrolling.
  await Promise.allSettled([
    loadSide('baseline', '/diag-vq45/bonsai_input.glb', 'canvas-baseline', 'status-baseline'),
    loadSide('vq45-no-prune', '/diag-vq45/wmv-vq45-no-prune.glb', 'canvas-compressed', 'status-compressed'),
  ]);
  log('--- done ---');
  // Set a sentinel for headless screenshot tools.
  document.body.setAttribute('data-diag-state', 'done');
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('run').addEventListener('click', () => {
    document.getElementById('log').textContent = '';
    main().catch((err) => log('main crashed:', err.message));
  });
  // Auto-run if `?auto=1` (used by the screenshot tool).
  if (new URLSearchParams(location.search).get('auto') === '1') {
    main().catch((err) => log('main crashed:', err.message));
  }
});
