/**
 * Minimal WebGL2 3D Gaussian Splatting renderer.
 *
 * Reference pattern: project each splat as an instanced quad sized by the
 * eigenvectors of its 2D projected covariance. Color is the splat's DC color;
 * alpha is `opacity * exp(-0.5 * x^T Σ^-1 x)` in screen space.
 *
 * Sorting: back-to-front by view-space depth (negative Z in camera space).
 * Computed in a Web Worker (`sort-worker.ts`) using a 16-bit counting sort.
 * The main thread renders with whatever sort is currently uploaded and swaps
 * in a fresh sort as soon as the worker posts one back — so orbiting never
 * blocks on the ~16 ms sort. If worker creation fails we fall back to the
 * same counting sort on the main thread.
 *
 * Math intentionally kept self-contained (no matrix lib) so the bundle stays
 * small for static hosting.
 */
import { type SplatScene, shRestCoefCount } from './splat-scene.js';
import { type CameraState, viewProjMatrix } from './camera.js';
import SortWorker from './sort-worker.js?worker';
import type { SortWorkerOutbound } from './sort-worker.js';

/** Texels per splat in the SH-rest texture (fits SH3: 45 floats → 12 RGBA-fp16 texels). */
const SH_TEXELS_PER_SPLAT = 12;
/** Splats-per-row in the SH-rest texture. Picked so width ≈ 4080 ≤ common max. */
const SH_SPLATS_PER_ROW = 340;
/** Texture width = SH_TEXELS_PER_SPLAT * SH_SPLATS_PER_ROW = 4080. */
const SH_TEX_WIDTH = SH_TEXELS_PER_SPLAT * SH_SPLATS_PER_ROW;

const VERT = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec2 a_quadCorner;     // unit quad in [-1,1]
layout(location = 1) in float a_splatIdx;      // ORIGINAL splat index (sorted by depth per frame)

uniform mat4 u_view;
uniform mat4 u_viewProj;
uniform vec2 u_viewport;
uniform vec3 u_camPos;
uniform vec2 u_focal;              // (fx, fy) in pixels
// Per-splat geometry — antimatter15-style packed RGBA32UI texture, 2 texels per splat.
//   even col (idx&0x3ff)<<1   → uvec4(posX_f32, posY_f32, posZ_f32, opacity_f32) via uintBitsToFloat
//   odd  col ((idx&0x3ff)<<1)|1 → uvec4(packHalf2x16(c00,c01),
//                                       packHalf2x16(c02,c11),
//                                       packHalf2x16(c12,c22),
//                                       0)
// Width is fixed at 2048 (1024 splats per row), height = ceil(N/1024).
uniform highp usampler2D u_packed;
// Color sampler stays a separate fp32 texture so the SH-rest ON/OFF toggle can
// hot-swap between raw f_dc (ON) and pre-baked clamped colorDC (OFF) without
// rewriting the packed geometry texture.
uniform sampler2D u_color;         // RGBA32F: (r, g, b, _)
uniform int u_colorTexWidth;       // width of color texture (= COLOR_TEX_WIDTH=2048)
// SH-rest texture (separate; has its own packed layout).
uniform sampler2D u_shRest;        // RGBA16F, [splat][k=0..11 texels], NEAREST
uniform int u_shDegree;            // 0 → no SH, 1/2/3 → eval bands
uniform int u_shSplatsPerRow;      // splats per row in SH texture
uniform int u_shTexelsPerSplat;    // texels per splat block (12 = SH3)

out vec2 v_offset;        // texel offset of this fragment from the splat center
out vec4 v_color;
out vec3 v_cov;           // (a, b, c) of 2D inv-cov

// Color texture is a 2D RGBA32F at width COLOR_TEX_WIDTH (1-texel per splat).
ivec2 colorTexCoord(int idx) {
  return ivec2(idx % u_colorTexWidth, idx / u_colorTexWidth);
}

// Inria 3DGS reference SH band constants.
const float SH_C0_CONST = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 =  1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 =  0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 =  0.5462742152960396;
const float SH_C3_0 = -0.5900435899266435;
const float SH_C3_1 =  2.890611442640554;
const float SH_C3_2 = -0.4570457994644658;
const float SH_C3_3 =  0.3731763325901154;
const float SH_C3_4 = -0.4570457994644658;
const float SH_C3_5 =  1.445305721320277;
const float SH_C3_6 = -0.5900435899266435;

// Fetch the k-th SH-rest coefficient (RGB triple) for splat \`splatIdx\`. Layout:
//   texelInSplat = floor(k * 3 / 4)        // 4 floats per RGBA texel
//   compInTexel  = (k * 3) mod 4
// The 3 RGB values may straddle two texels; we sample whichever it lands in.
vec3 fetchShRestK(int splatIdx, int k) {
  int floatIdx = k * 3;
  int texel0 = floatIdx >> 2;          // floor(floatIdx / 4)
  int comp0  = floatIdx & 3;           // floatIdx mod 4
  int splatRow = splatIdx / u_shSplatsPerRow;
  int splatCol = splatIdx - splatRow * u_shSplatsPerRow;
  int colBase = splatCol * u_shTexelsPerSplat;
  // Texel 0:
  int t0col = colBase + texel0;
  vec4 a = texelFetch(u_shRest, ivec2(t0col, splatRow), 0);
  // Possibly texel 1 (if RGB straddles a 4-float boundary):
  if (comp0 <= 1) {
    // All 3 fit in this texel starting at comp0.
    if (comp0 == 0) return a.rgb;
    return a.gba; // comp0 == 1
  }
  // Straddle: 1 or 2 components in this texel, rest in next.
  int t1col = colBase + texel0 + 1;
  vec4 b = texelFetch(u_shRest, ivec2(t1col, splatRow), 0);
  if (comp0 == 2) {
    // a.b a.a b.r
    return vec3(a.b, a.a, b.r);
  }
  // comp0 == 3:
  return vec3(a.a, b.r, b.g);
}

vec3 evalShRest(int splatIdx, vec3 dir) {
  if (u_shDegree <= 0) return vec3(0.0);
  float x = dir.x, y = dir.y, z = dir.z;
  vec3 col = vec3(0.0);
  // l=1 (k = 0..2):
  col += SH_C1 * (-y) * fetchShRestK(splatIdx, 0);
  col += SH_C1 * ( z) * fetchShRestK(splatIdx, 1);
  col += SH_C1 * (-x) * fetchShRestK(splatIdx, 2);
  if (u_shDegree < 2) return col;
  float xx = x*x, yy = y*y, zz = z*z;
  float xy = x*y, yz = y*z, xz = x*z;
  // l=2 (k = 3..7):
  col += SH_C2_0 * (xy)               * fetchShRestK(splatIdx, 3);
  col += SH_C2_1 * (yz)               * fetchShRestK(splatIdx, 4);
  col += SH_C2_2 * (2.0*zz - xx - yy) * fetchShRestK(splatIdx, 5);
  col += SH_C2_3 * (xz)               * fetchShRestK(splatIdx, 6);
  col += SH_C2_4 * (xx - yy)          * fetchShRestK(splatIdx, 7);
  if (u_shDegree < 3) return col;
  // l=3 (k = 8..14):
  col += SH_C3_0 * y * (3.0*xx - yy)       * fetchShRestK(splatIdx, 8);
  col += SH_C3_1 * xy * z                  * fetchShRestK(splatIdx, 9);
  col += SH_C3_2 * y * (4.0*zz - xx - yy)  * fetchShRestK(splatIdx, 10);
  col += SH_C3_3 * z * (2.0*zz - 3.0*xx - 3.0*yy) * fetchShRestK(splatIdx, 11);
  col += SH_C3_4 * x * (4.0*zz - xx - yy)  * fetchShRestK(splatIdx, 12);
  col += SH_C3_5 * z * (xx - yy)           * fetchShRestK(splatIdx, 13);
  col += SH_C3_6 * x * (xx - 3.0*yy)       * fetchShRestK(splatIdx, 14);
  return col;
}

void main() {
  // Stage 6 packed-fetch layout: 2 RGBA32UI texels per splat at width 2048.
  //   col0 = (sIdx & 0x3ff) << 1, row = sIdx >> 10  → pos+opacity (4 floats)
  //   col1 = col0 | 1                              → cov3d as 6 halves (8th slot reserved)
  int sIdx = int(a_splatIdx);
  int packRow = sIdx >> 10;
  int packCol0 = (sIdx & 0x3ff) << 1;
  uvec4 pkPos = texelFetch(u_packed, ivec2(packCol0, packRow), 0);
  uvec4 pkCov = texelFetch(u_packed, ivec2(packCol0 | 1, packRow), 0);
  vec3 a_pos = uintBitsToFloat(pkPos.xyz);
  float opacity = uintBitsToFloat(pkPos.w);
  // Cov3d unpacked from 3× packHalf2x16 → 6 floats. Stored unscaled (no ×4),
  // matching the existing pipeline's covariance values.
  vec2 h0 = unpackHalf2x16(pkCov.x);  // (c00, c01)
  vec2 h1 = unpackHalf2x16(pkCov.y);  // (c02, c11)
  vec2 h2 = unpackHalf2x16(pkCov.z);  // (c12, c22)
  // Color sampled from its own RGBA32F texture so the SH-rest ON/OFF swap can
  // hot-swap between raw f_dc and pre-baked colorDC without rebuilding pack.
  vec4 colorRgba = texelFetch(u_color, colorTexCoord(sIdx), 0);
  vec4 a_color = vec4(colorRgba.rgb, opacity);

  // View-space splat center.
  vec4 viewPos4 = u_view * vec4(a_pos, 1.0);
  float vx = viewPos4.x, vy = viewPos4.y, vz = viewPos4.z;
  // Behind camera (or zero depth): collapse off-screen.
  if (vz >= -1e-4) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_offset = vec2(0.0);
    v_color = vec4(0.0);
    v_cov = vec3(1.0, 0.0, 1.0);
    return;
  }

  // Stage 7: frustum cull BEFORE the heavy J·Σ·J^T projection. Saves the
  // 9 mat-muls + 2x2 eigen for any splat whose clip-space center sits well
  // outside the [-w, w] box. The 1.2 margin (~20% of frustum) is wider than
  // strictly needed — it accounts for the projected splat radius without
  // having to compute it first. antimatter15 uses the same idiom.
  vec4 clipCenter = u_viewProj * vec4(a_pos, 1.0);
  float clipMargin = 1.2 * clipCenter.w;
  if (clipCenter.x < -clipMargin || clipCenter.x > clipMargin
   || clipCenter.y < -clipMargin || clipCenter.y > clipMargin
   || clipCenter.z < -clipCenter.w || clipCenter.z > clipCenter.w) {
    gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
    v_offset = vec2(0.0);
    v_color = vec4(0.0);
    v_cov = vec3(1.0, 0.0, 1.0);
    return;
  }

  // World 3D covariance (symmetric, upper triangle).
  float c00 = h0.x, c01 = h0.y;
  float c02 = h1.x, c11 = h1.y;
  float c12 = h2.x, c22 = h2.y;

  // R = upper-left 3x3 of u_view (column-major in WebGL).
  float rx0 = u_view[0].x, rx1 = u_view[1].x, rx2 = u_view[2].x;
  float ry0 = u_view[0].y, ry1 = u_view[1].y, ry2 = u_view[2].y;
  float rz0 = u_view[0].z, rz1 = u_view[1].z, rz2 = u_view[2].z;

  // T = R * Σ_w  (3x3).
  float t00 = rx0 * c00 + rx1 * c01 + rx2 * c02;
  float t01 = rx0 * c01 + rx1 * c11 + rx2 * c12;
  float t02 = rx0 * c02 + rx1 * c12 + rx2 * c22;
  float t10 = ry0 * c00 + ry1 * c01 + ry2 * c02;
  float t11 = ry0 * c01 + ry1 * c11 + ry2 * c12;
  float t12 = ry0 * c02 + ry1 * c12 + ry2 * c22;
  float t20 = rz0 * c00 + rz1 * c01 + rz2 * c02;
  float t21 = rz0 * c01 + rz1 * c11 + rz2 * c12;
  float t22 = rz0 * c02 + rz1 * c12 + rz2 * c22;

  // Σ_v = T * R^T (symmetric upper tri).
  float v00 = t00 * rx0 + t01 * rx1 + t02 * rx2;
  float v01 = t00 * ry0 + t01 * ry1 + t02 * ry2;
  float v02 = t00 * rz0 + t01 * rz1 + t02 * rz2;
  float v11 = t10 * ry0 + t11 * ry1 + t12 * ry2;
  float v12 = t10 * rz0 + t11 * rz1 + t12 * rz2;
  float v22 = t20 * rz0 + t21 * rz1 + t22 * rz2;

  // Image-plane Jacobian for the perspective projection.
  float fx = u_focal.x, fy = u_focal.y;
  float invVz = 1.0 / vz;
  float invVz2 = invVz * invVz;
  float j00 = -fx * invVz;
  float j02 =  fx * vx * invVz2;
  float j11 = -fy * invVz;
  float j12 =  fy * vy * invVz2;

  // Σ_2D = J * Σ_v * J^T  (2x2 symmetric).
  float m00 = j00 * v00 + j02 * v02;
  float m01 = j00 * v01 + j02 * v12;
  float m02 = j00 * v02 + j02 * v22;
  float m11 = j11 * v11 + j12 * v12;
  float m12 = j11 * v12 + j12 * v22;
  float s00 = m00 * j00 + m02 * j02;
  float s01 = m01 * j11 + m02 * j12;
  float s11 = m11 * j11 + m12 * j12;
  // Sub-pixel dilation.
  float a = s00 + 0.3;
  float b = s01;
  float c = s11 + 0.3;
  float det = a * c - b * b;
  if (det <= 0.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    v_offset = vec2(0.0);
    v_color = vec4(0.0);
    v_cov = vec3(1.0, 0.0, 1.0);
    return;
  }
  float invDet = 1.0 / det;
  float ia = c * invDet;
  float ib = -b * invDet;
  float ic = a * invDet;
  // Pixel radius from larger eigenvalue.
  float tr = (a + c) * 0.5;
  float sub = (a - c) * 0.5;
  float lambdaMax = tr + sqrt(sub * sub + b * b);
  // Stage 7b: shrink quad radius 3σ → 2.5σ. At 2.5σ a Gaussian still
  // captures ~98.8% of its mass (3σ is 99.7%); the visible difference on
  // dense scenes is negligible but the per-quad area drops ~30%, directly
  // cutting fillrate-bound fragment-shader work. antimatter15 uses 2.0σ
  // (more aggressive); 2.5σ is the conservative middle ground.
  float r = 2.5 * sqrt(max(lambdaMax, 1e-6));
  float rMax = max(u_viewport.x, u_viewport.y) * 0.5;
  float rPx = min(r, rMax);

  // Projected NDC center (reusing the clip-space center computed for the
  // Stage 7 frustum cull above — no second mat4 multiply).
  vec3 ndc = clipCenter.xyz / clipCenter.w;
  vec2 offsetPx = a_quadCorner * rPx;
  vec2 offsetNdc = offsetPx / u_viewport * 2.0;
  gl_Position = vec4(ndc.xy + offsetNdc, ndc.z, 1.0);
  v_offset = offsetPx;

  // Color reconstruction.
  vec3 finalRgb;
  if (u_shDegree > 0) {
    vec3 viewDir = normalize(a_pos - u_camPos);
    vec3 shCol = evalShRest(int(a_splatIdx), viewDir);
    finalRgb = clamp(0.5 + SH_C0_CONST * a_color.rgb + shCol, 0.0, 1.0);
  } else {
    finalRgb = a_color.rgb;
  }
  v_color = vec4(finalRgb, a_color.a);
  v_cov = vec3(ia, ib, ic);
}
`;

const FRAG = /* glsl */ `#version 300 es
precision highp float;
in vec2 v_offset;
in vec4 v_color;
in vec3 v_cov;            // inverse 2D cov (a, b, c)

out vec4 fragColor;

void main() {
  // Evaluate Gaussian in screen space:  -0.5 * x^T Σ^-1 x
  //   where Σ^-1 = [[a, b],[b, c]]
  float x = v_offset.x;
  float y = v_offset.y;
  float p = -0.5 * (v_cov.x * x * x + v_cov.z * y * y) - v_cov.y * x * y;
  if (p < -8.0) discard;     // very-low-alpha cutoff
  float alpha = v_color.a * exp(p);
  if (alpha < 1.0 / 255.0) discard;
  // Premultiplied output: blendFunc(ONE, ONE_MINUS_SRC_ALPHA).
  fragColor = vec4(v_color.rgb * alpha, alpha);
}
`;

function compile(gl: WebGL2RenderingContext, src: string, kind: number): WebGLShader {
  const sh = gl.createShader(kind);
  if (!sh) throw new Error('createShader failed');
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error(`shader compile: ${log}`);
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: WebGLShader, fs: WebGLShader): WebGLProgram {
  const p = gl.createProgram();
  if (!p) throw new Error('createProgram failed');
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error(`program link: ${log}`);
  }
  return p;
}

/** Texture width for the color texture (1 texel per splat). */
const COLOR_TEX_WIDTH = 2048;
/** Stage 6 packed-texture width. Matches antimatter15 (1024 splats × 2 texels = 2048). */
const PACKED_TEX_WIDTH = 2048;
/** Splats per row in the packed texture (PACKED_TEX_WIDTH / 2 texels per splat). */
const PACKED_SPLATS_PER_ROW = 1024;

export class SplatRenderer {
  private gl!: WebGL2RenderingContext;
  private program!: WebGLProgram;
  private vao!: WebGLVertexArrayObject;
  private quadBuf!: WebGLBuffer;
  /** Per-splat ORIGINAL splat index (float32) in sort order. The ONLY
   *  per-frame buffer upload. */
  private idxBuf!: WebGLBuffer;

  /** Per-splat static data textures (uploaded once at scene load). */
  // Stage 6: packed RGBA32UI texture — replaces posOpaTex + cov3dLoTex + cov3dHiTex.
  private packedTex: WebGLTexture | null = null;
  private packedTexHeight = 0;
  // Color textures stay separate so the SH-rest ON/OFF toggle can hot-swap
  // between raw f_dc (ON) and pre-baked colorDC (OFF) without rebuilding pack.
  private colorTex: WebGLTexture | null = null;        // Raw f_dc if SH-rest is on; baked colorDC otherwise.
  private colorBakedTex: WebGLTexture | null = null;   // Always the baked colorDC (used when SH-rest is toggled off).
  private colorTexHeight = 0;

  /** Runtime toggle: skip SH-rest evaluation in the vertex shader (~3x perf win). */
  private shRestEnabled = true;

  /** RGBA16F texture holding SH-rest coefficients. */
  private shTex: WebGLTexture | null = null;
  private shTexHeight = 0;

  private uView!: WebGLUniformLocation;
  private uViewProj!: WebGLUniformLocation;
  private uViewport!: WebGLUniformLocation;
  private uCamPos!: WebGLUniformLocation;
  private uFocal!: WebGLUniformLocation;
  private uPacked!: WebGLUniformLocation;
  private uColor!: WebGLUniformLocation;
  private uColorTexWidth!: WebGLUniformLocation;
  private uShRest!: WebGLUniformLocation;
  private uShDegree!: WebGLUniformLocation;
  private uShSplatsPerRow!: WebGLUniformLocation;
  private uShTexelsPerSplat!: WebGLUniformLocation;

  private scene: SplatScene | null = null;

  // Per-frame: just the sort index buffer (ORIGINAL idx in sort order).
  // `idxData` is what's currently uploaded to the GPU; `pendingIdx` is the
  // most recently received-but-not-yet-uploaded sort from the worker.
  private idxData: Float32Array | null = null;
  private pendingIdx: Float32Array | null = null;

  // Main-thread fallback scratch (only used when the worker fails to spawn).
  private depths: Float32Array | null = null;
  private indices: Uint32Array | null = null;
  private sortBucketOf: Uint32Array | null = null;
  private sortOut: Uint32Array | null = null;

  // Sort throttling.
  private lastSortCameraKey = '';

  // Worker-sort plumbing.
  private sortWorker: Worker | null = null;
  /** requestId of the most recent sort we KICKED OFF. */
  private nextSortRequestId = 0;
  /** requestId of the most recent sort we APPLIED to the GPU. Drops staler results. */
  private lastAppliedRequestId = -1;
  /** True if the worker is currently sorting a request we sent. */
  private sortInFlight = false;
  /** If a new sort key arrives while a sort is in flight, stash the view here and dispatch on completion. */
  private deferredView: Float32Array | null = null;
  private deferredRequestId = -1;

  init(canvas: HTMLCanvasElement): void {
    const gl = canvas.getContext('webgl2', {
      premultipliedAlpha: false,
      antialias: false,
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;

    const vs = compile(gl, VERT, gl.VERTEX_SHADER);
    const fs = compile(gl, FRAG, gl.FRAGMENT_SHADER);
    this.program = link(gl, vs, fs);
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    this.uView = gl.getUniformLocation(this.program, 'u_view')!;
    this.uViewProj = gl.getUniformLocation(this.program, 'u_viewProj')!;
    this.uViewport = gl.getUniformLocation(this.program, 'u_viewport')!;
    this.uCamPos = gl.getUniformLocation(this.program, 'u_camPos')!;
    this.uFocal = gl.getUniformLocation(this.program, 'u_focal')!;
    this.uPacked = gl.getUniformLocation(this.program, 'u_packed')!;
    this.uColor = gl.getUniformLocation(this.program, 'u_color')!;
    this.uColorTexWidth = gl.getUniformLocation(this.program, 'u_colorTexWidth')!;
    this.uShRest = gl.getUniformLocation(this.program, 'u_shRest')!;
    this.uShDegree = gl.getUniformLocation(this.program, 'u_shDegree')!;
    this.uShSplatsPerRow = gl.getUniformLocation(this.program, 'u_shSplatsPerRow')!;
    this.uShTexelsPerSplat = gl.getUniformLocation(this.program, 'u_shTexelsPerSplat')!;

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);

    // Quad corners in [-1,1].
    this.quadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      1, -1, 1, 1, -1, 1,
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Only one per-instance attribute now: the sorted splat index.
    this.idxBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(1, 1);

    // Float-color-buffer extension needed for RGBA32F. Most desktop/iOS Safari
    // expose it. Without it we'd need to fall back to RGBA16F + manual encode.
    if (!gl.getExtension('EXT_color_buffer_float')) {
      // Not strictly required for sampling — only for FBO render targets —
      // but flagging in case a driver complains during texImage2D.
      console.warn('EXT_color_buffer_float not available; RGBA32F textures may still work for sampling.');
    }

    gl.bindVertexArray(null);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    // Premultiplied alpha back-to-front composit.
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    // Spawn the sort worker. Failure is non-fatal — we'll fall back to a
    // main-thread counting sort (same algorithm) in render().
    try {
      this.sortWorker = new SortWorker();
      this.sortWorker.addEventListener('message', (ev: MessageEvent<SortWorkerOutbound>) => {
        const msg = ev.data;
        if (msg.type === 'sorted') {
          this.onSortResult(msg.requestId, msg.indices);
        }
      });
      this.sortWorker.addEventListener('error', (e) => {
        console.warn('sort worker error; falling back to main-thread sort:', e.message);
        this.sortWorker?.terminate();
        this.sortWorker = null;
      });
    } catch (e) {
      console.warn('failed to spawn sort worker; using main-thread sort:', (e as Error).message);
      this.sortWorker = null;
    }
  }

  /** Receive a sort result from the worker. Drop stale ones; stage fresh ones for upload on the next render. */
  private onSortResult(requestId: number, indices: Float32Array): void {
    this.sortInFlight = false;
    if (requestId <= this.lastAppliedRequestId) {
      // Stale — a newer sort already landed. Drop.
    } else {
      this.pendingIdx = indices;
      this.lastAppliedRequestId = requestId;
    }
    // If a newer view arrived while we were busy, kick that off now.
    if (this.deferredView && this.sortWorker) {
      const view = this.deferredView;
      const reqId = this.deferredRequestId;
      this.deferredView = null;
      this.deferredRequestId = -1;
      this.sortInFlight = true;
      this.sortWorker.postMessage({ type: 'sort', requestId: reqId, view }, [view.buffer]);
    }
  }

  setScene(scene: SplatScene): void {
    const gl = this.gl;
    this.scene = scene;
    const N = scene.count;
    this.idxData = new Float32Array(N);
    this.depths = new Float32Array(N);
    this.indices = new Uint32Array(N);
    this.sortBucketOf = new Uint32Array(N);
    this.sortOut = new Uint32Array(N);
    for (let i = 0; i < N; i++) this.indices[i] = i;

    // Upload SH-rest into a 2D RGBA16F texture, padded to SH3 (12 texels/splat).
    if (this.shTex) {
      gl.deleteTexture(this.shTex);
      this.shTex = null;
      this.shTexHeight = 0;
    }
    if (scene.shRest && scene.shDegree && scene.shDegree > 0) {
      const coefCount = shRestCoefCount(scene.shDegree);             // 3/8/15
      const splatsPerRow = SH_SPLATS_PER_ROW;
      const height = Math.ceil(N / splatsPerRow);
      // 4 floats per RGBA texel × SH_TEXELS_PER_SPLAT texels per splat.
      const floatsPerSplat = SH_TEXELS_PER_SPLAT * 4;                // 48
      const totalFloats = SH_TEX_WIDTH * height * 4;
      const packed = new Float32Array(totalFloats);
      // Layout per splat row block: [splatCol=0..splatsPerRow-1] each block is
      // 12 RGBA texels = 48 floats. We pack 45 floats (or less for lower
      // degrees) as RGBA quartets; trailing floats stay zero.
      for (let i = 0; i < N; i++) {
        const splatRow = Math.floor(i / splatsPerRow);
        const splatCol = i - splatRow * splatsPerRow;
        const dstBase = splatRow * SH_TEX_WIDTH * 4 + splatCol * floatsPerSplat;
        const srcBase = i * coefCount * 3;
        // Copy 3*coefCount floats; padding (e.g., 45 → 48) stays zero.
        for (let f = 0; f < coefCount * 3; f++) {
          packed[dstBase + f] = scene.shRest[srcBase + f];
        }
      }
      // Float32 → Float16Array via manual conversion (no native Float16 in JS yet).
      const fp16 = new Uint16Array(totalFloats);
      for (let i = 0; i < totalFloats; i++) fp16[i] = toHalf(packed[i]);
      this.shTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.shTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, SH_TEX_WIDTH, height, 0,
        gl.RGBA, gl.HALF_FLOAT, fp16);
      this.shTexHeight = height;
      void this.shTexHeight;
      gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // Stage 6: build the packed RGBA32UI geometry texture (2 texels per splat).
    // Layout:
    //   even col (splatCol*2)   = uvec4(posX_f32, posY_f32, posZ_f32, opacity_f32) via floatBitsToUint
    //   odd  col (splatCol*2+1) = uvec4(packHalf2x16(c00,c01),
    //                                   packHalf2x16(c02,c11),
    //                                   packHalf2x16(c12,c22),
    //                                   0)  // 8th half-slot reserved
    // Texture width fixed at PACKED_TEX_WIDTH=2048, so each row holds
    // PACKED_SPLATS_PER_ROW=1024 splats. Height = ceil(N / 1024).
    const packedTexHeight = Math.ceil(N / PACKED_SPLATS_PER_ROW);
    const packedTexels = PACKED_TEX_WIDTH * packedTexHeight;
    const packed = new Uint32Array(packedTexels * 4);
    const packedF32 = new Float32Array(packed.buffer);   // for floatBitsToUint via aliased view

    // Color textures stay separate (1 texel per splat) so SH-rest ON/OFF can swap source.
    const colorTexHeight = Math.ceil(N / COLOR_TEX_WIDTH);
    const colorPadded = COLOR_TEX_WIDTH * colorTexHeight;
    const colorPacked = new Float32Array(colorPadded * 4);
    const colorBaked = new Float32Array(colorPadded * 4);

    const useRaw = !!scene.shRest && !!scene.dcRaw && !!scene.shDegree && scene.shDegree > 0;
    const col = useRaw ? scene.dcRaw! : scene.colorDC;
    for (let i = 0; i < N; i++) {
      colorBaked[i * 4 + 0] = scene.colorDC[i * 3 + 0];
      colorBaked[i * 4 + 1] = scene.colorDC[i * 3 + 1];
      colorBaked[i * 4 + 2] = scene.colorDC[i * 3 + 2];
      colorPacked[i * 4 + 0] = col[i * 3 + 0];
      colorPacked[i * 4 + 1] = col[i * 3 + 1];
      colorPacked[i * 4 + 2] = col[i * 3 + 2];
    }

    for (let i = 0; i < N; i++) {
      const splatRow = i >>> 10;                       // i / 1024
      const splatCol = i & 0x3ff;                      // i % 1024
      // Base offset (in uint32 units) of texel-0 for this splat.
      const texel0Base = (splatRow * PACKED_TEX_WIDTH + (splatCol << 1)) * 4;
      const texel1Base = texel0Base + 4;

      // Texel 0: position + opacity as 4 float32 (bit-reinterpreted as uint).
      packedF32[texel0Base + 0] = scene.positions[i * 3 + 0];
      packedF32[texel0Base + 1] = scene.positions[i * 3 + 1];
      packedF32[texel0Base + 2] = scene.positions[i * 3 + 2];
      packedF32[texel0Base + 3] = scene.opacity[i];

      // 3D world covariance (symmetric upper triangle) from quaternion + log-scale.
      const sx = Math.exp(scene.scales[i * 3 + 0]);
      const sy = Math.exp(scene.scales[i * 3 + 1]);
      const sz = Math.exp(scene.scales[i * 3 + 2]);
      const x = scene.rotations[i * 4 + 0];
      const y = scene.rotations[i * 4 + 1];
      const z = scene.rotations[i * 4 + 2];
      const w = scene.rotations[i * 4 + 3];
      const xx = x * x, yy = y * y, zz = z * z;
      const xy = x * y, xz = x * z, yz = y * z;
      const wx = w * x, wy = w * y, wz = w * z;
      const r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz),     r02 = 2 * (xz + wy);
      const r10 = 2 * (xy + wz),     r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
      const r20 = 2 * (xz - wy),     r21 = 2 * (yz + wx),     r22 = 1 - 2 * (xx + yy);
      const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
      const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
      const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
      const c00 = m00 * m00 + m01 * m01 + m02 * m02;
      const c01 = m00 * m10 + m01 * m11 + m02 * m12;
      const c02 = m00 * m20 + m01 * m21 + m02 * m22;
      const c11 = m10 * m10 + m11 * m11 + m12 * m12;
      const c12 = m10 * m20 + m11 * m21 + m12 * m22;
      const c22 = m20 * m20 + m21 * m21 + m22 * m22;

      // Texel 1: cov3d packed as 3× packHalf2x16. Pairs match the shader's
      // (h0=(c00,c01), h1=(c02,c11), h2=(c12,c22)) unpack order.
      packed[texel1Base + 0] = packHalf2x16(c00, c01);
      packed[texel1Base + 1] = packHalf2x16(c02, c11);
      packed[texel1Base + 2] = packHalf2x16(c12, c22);
      packed[texel1Base + 3] = 0;
    }

    if (this.packedTex) gl.deleteTexture(this.packedTex);
    if (this.colorTex) gl.deleteTexture(this.colorTex);
    if (this.colorBakedTex) gl.deleteTexture(this.colorBakedTex);
    this.packedTex = makeUint32RGBATex(gl, PACKED_TEX_WIDTH, packedTexHeight, packed);
    this.packedTexHeight = packedTexHeight;
    void this.packedTexHeight;
    this.colorTex = makeFloat32Tex(gl, COLOR_TEX_WIDTH, colorTexHeight, colorPacked);
    this.colorBakedTex = makeFloat32Tex(gl, COLOR_TEX_WIDTH, colorTexHeight, colorBaked);
    this.colorTexHeight = colorTexHeight;
    void this.colorTexHeight;

    this.lastSortCameraKey = '';
    this.pendingIdx = null;
    this.nextSortRequestId = 0;
    this.lastAppliedRequestId = -1;
    this.sortInFlight = false;
    this.deferredView = null;
    this.deferredRequestId = -1;

    // Seed the GPU index buffer with an identity order so the very first frame
    // (before any sort completes) renders SOMETHING instead of N copies of
    // splat 0. The first real sort from the worker (or main-thread fallback)
    // will overwrite this within one frame.
    for (let i = 0; i < N; i++) this.idxData[i] = i;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.idxData, gl.DYNAMIC_DRAW);

    // Hand a COPY of positions to the worker (transferred zero-copy). We copy
    // because the renderer reuses the original positions for fallback sorting
    // and bbox / scene-positions providers in main.ts.
    if (this.sortWorker) {
      const posCopy = new Float32Array(scene.positions);
      this.sortWorker.postMessage({ type: 'init', positions: posCopy }, [posCopy.buffer]);
    }
  }

  /**
   * Render one frame for `camera` at canvas `width`x`height` (CSS pixels are
   * scaled to backing-store pixels by the caller).
   *
   * `width`/`height` are backing-store dimensions (gl.drawingBufferWidth).
   */
  render(camera: CameraState, width: number, height: number): void {
    const gl = this.gl;
    if (!this.scene || !this.idxData || !this.depths || !this.indices
      || !this.packedTex || !this.colorTex || !this.colorBakedTex) {
      gl.clearColor(0.02, 0.024, 0.032, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      return;
    }
    const N = this.scene.count;
    const vp = viewProjMatrix(camera, width / Math.max(1, height));
    const view = vp.view;

    // Sort-trigger key: quantized camera state. When this changes we want a
    // new sort. With the worker that means "ask the worker"; without it we
    // run the sort inline (legacy main-thread fallback).
    const key = `${Math.round(camera.yaw * 16)}.${Math.round(camera.pitch * 16)}.${Math.round(camera.distance * 8)}.${Math.round(camera.target[0] * 4)}.${Math.round(camera.target[1] * 4)}.${Math.round(camera.target[2] * 4)}`;
    const sortKeyChanged = key !== this.lastSortCameraKey;

    if (this.sortWorker) {
      // ---- Worker path ----
      // Kick off a new sort when the camera key changes. If a sort is already
      // in flight, stash the latest view; we'll dispatch it on completion so
      // we always end up sorting the most-recent camera.
      if (sortKeyChanged) {
        this.lastSortCameraKey = key;
        const viewCopy = new Float32Array(view);  // worker takes ownership
        const reqId = ++this.nextSortRequestId;
        if (this.sortInFlight) {
          // Replace any older deferred request — only the newest view matters.
          this.deferredView = viewCopy;
          this.deferredRequestId = reqId;
        } else {
          this.sortInFlight = true;
          this.sortWorker.postMessage({ type: 'sort', requestId: reqId, view: viewCopy }, [viewCopy.buffer]);
        }
      }
      // Upload a freshly-arrived sort to the GPU (one frame's latency vs the
      // camera, but no main-thread hitch).
      if (this.pendingIdx && this.pendingIdx.length === N) {
        this.idxData = this.pendingIdx;
        this.pendingIdx = null;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
        gl.bufferData(gl.ARRAY_BUFFER, this.idxData, gl.DYNAMIC_DRAW);
      }
    } else {
      // ---- Main-thread fallback ----
      // Lazily allocate fallback scratch (worker path never touches these).
      if (!this.depths || this.depths.length !== N) {
        this.depths = new Float32Array(N);
        this.indices = new Uint32Array(N);
        this.sortBucketOf = new Uint32Array(N);
        this.sortOut = new Uint32Array(N);
        for (let i = 0; i < N; i++) this.indices[i] = i;
      }
      if (sortKeyChanged) {
        const depths = this.depths;
        const idx = this.indices!;
        const pos = this.scene.positions;
        for (let i = 0; i < N; i++) {
          const x = pos[i * 3 + 0];
          const y = pos[i * 3 + 1];
          const z = pos[i * 3 + 2];
          const vz = view[2] * x + view[6] * y + view[10] * z + view[14];
          depths[i] = vz;
        }
        sortIndicesByDepth(idx, depths, this.sortBucketOf!, this.sortOut!);
        const idxOut = this.idxData!;
        for (let n = 0; n < N; n++) idxOut[n] = idx[n];
        gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
        gl.bufferData(gl.ARRAY_BUFFER, idxOut, gl.DYNAMIC_DRAW);
        this.lastSortCameraKey = key;
      }
    }
    const fy = (height * 0.5) / Math.tan(camera.fovYRad * 0.5);
    const fx = fy;  // square pixels

    gl.viewport(0, 0, width, height);
    gl.clearColor(0.02, 0.024, 0.032, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniformMatrix4fv(this.uViewProj, false, vp.viewProj);
    gl.uniform2f(this.uViewport, width, height);
    gl.uniform3f(this.uCamPos, vp.eye[0], vp.eye[1], vp.eye[2]);
    gl.uniform2f(this.uFocal, fx, fy);
    gl.uniform1i(this.uColorTexWidth, COLOR_TEX_WIDTH);
    const shDeg = (this.shRestEnabled && this.shTex && this.scene.shDegree) ? this.scene.shDegree : 0;
    gl.uniform1i(this.uShDegree, shDeg);
    gl.uniform1i(this.uShSplatsPerRow, SH_SPLATS_PER_ROW);
    gl.uniform1i(this.uShTexelsPerSplat, SH_TEXELS_PER_SPLAT);

    // Bind packed RGBA32UI on TEXTURE0, color on TEXTURE1, SH-rest on TEXTURE2.
    // When SH-rest is toggled off, swap to the baked-color texture so the shader's
    // "just pass a_color.rgb" branch gets pre-clamped display values.
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.packedTex);
    gl.uniform1i(this.uPacked, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, shDeg > 0 ? this.colorTex : this.colorBakedTex);
    gl.uniform1i(this.uColor, 1);
    gl.activeTexture(gl.TEXTURE2);
    if (this.shTex) {
      gl.bindTexture(gl.TEXTURE_2D, this.shTex);
    } else {
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    gl.uniform1i(this.uShRest, 2);

    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, N);
    gl.bindVertexArray(null);
  }

  /** Toggle SH-rest evaluation. When off, the renderer uses the pre-baked
   *  colorDC texture and skips the ~30 texelFetch/vertex SH-rest math —
   *  typically ~3x perf gain on dense scenes at the cost of view-dependent shading. */
  setShRestEnabled(enabled: boolean): void {
    this.shRestEnabled = enabled;
  }

  isShRestEnabled(): boolean {
    return this.shRestEnabled;
  }

  dispose(): void {
    if (this.sortWorker) {
      this.sortWorker.terminate();
      this.sortWorker = null;
    }
    const gl = this.gl;
    if (!gl) return;
    gl.deleteBuffer(this.idxBuf);
    gl.deleteBuffer(this.quadBuf);
    if (this.packedTex) gl.deleteTexture(this.packedTex);
    if (this.colorTex) gl.deleteTexture(this.colorTex);
    if (this.colorBakedTex) gl.deleteTexture(this.colorBakedTex);
    if (this.shTex) gl.deleteTexture(this.shTex);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
  }
}

/** Create + populate an RGBA32UI texture (Stage 6 packed geometry). NEAREST filter. */
function makeUint32RGBATex(gl: WebGL2RenderingContext, width: number, height: number, data: Uint32Array): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error('createTexture failed');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  // Integer textures REQUIRE NEAREST filtering on both axes per WebGL2 spec.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, width, height, 0,
    gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

/** GLSL `packHalf2x16` equivalent in JS — packs two floats into one uint32
 *  where the low 16 bits hold f16(x) and the high 16 bits hold f16(y). */
function packHalf2x16(x: number, y: number): number {
  return ((toHalf(x) | (toHalf(y) << 16)) >>> 0);
}

/** Create + populate an RGBA32F texture for per-splat data. NEAREST filter. */
function makeFloat32Tex(gl: WebGL2RenderingContext, width: number, height: number, data: Float32Array): WebGLTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error('createTexture failed');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

/** IEEE-754 single → half (binary16) conversion. Returns the 16-bit pattern. */
function toHalf(val: number): number {
  // Standard branch-free conversion borrowed from the canonical reference
  // (Mike Acton / OpenEXR ImfHalf). Avoids a DataView round-trip per value.
  if (!Number.isFinite(val)) {
    if (Number.isNaN(val)) return 0x7e00;
    return val < 0 ? 0xfc00 : 0x7c00;
  }
  if (val === 0) return Object.is(val, -0) ? 0x8000 : 0x0000;
  const sign = val < 0 ? 0x8000 : 0;
  const av = Math.abs(val);
  if (av >= 65504) return sign | 0x7bff;          // clamp to max half
  if (av < 6.103515625e-5) {
    // Subnormal half.
    const f = av / 6.103515625e-5;
    const m = Math.floor(f * 1024 + 0.5);
    return sign | (m & 0x3ff);
  }
  // Normal: extract exponent + mantissa via Float32 bit pattern.
  // We use a Float32Array view; the (slightly slower) but allocation-free
  // approach is acceptable here as this only runs at scene load.
  HALF_F32[0] = val;
  const bits = HALF_U32[0];
  const e = ((bits >> 23) & 0xff) - 127 + 15;
  const m = (bits >> 13) & 0x3ff;
  return sign | (e << 10) | m;
}
const HALF_F32 = new Float32Array(1);
const HALF_U32 = new Uint32Array(HALF_F32.buffer);

/**
 * Sort `idx` in place by `depths[idx[k]]` ascending (most-negative first =
 * farthest from camera first).
 *
 * 16-bit counting sort over the depth range. O(N + 65536), no boxed-Number
 * comparator overhead, no Array allocation. ~15× faster than Array.sort with
 * a comparator at N=1.24M (250ms → ~16ms in measured browser benches).
 *
 * Allocates two scratch Uint32Array(N) on each call. The renderer caches them
 * via `getSortScratch` to avoid reallocation churn.
 */
function sortIndicesByDepth(
  idx: Uint32Array,
  depths: Float32Array,
  bucketOf: Uint32Array,
  out: Uint32Array,
): void {
  const N = idx.length;
  if (N === 0) return;
  let dmin = Infinity, dmax = -Infinity;
  for (let i = 0; i < N; i++) {
    const d = depths[i];
    if (d < dmin) dmin = d;
    if (d > dmax) dmax = d;
  }
  const range = dmax - dmin;
  if (range < 1e-9) return;  // degenerate: all coplanar
  const NBUCKETS = 65536;
  const counts = new Uint32Array(NBUCKETS);
  const inv = (NBUCKETS - 1) / range;
  for (let i = 0; i < N; i++) {
    const b = ((depths[i] - dmin) * inv) | 0;
    bucketOf[i] = b;
    counts[b]++;
  }
  // Exclusive prefix sum.
  let total = 0;
  for (let b = 0; b < NBUCKETS; b++) {
    const c = counts[b];
    counts[b] = total;
    total += c;
  }
  // Place into output by stable bucket order.
  for (let i = 0; i < N; i++) {
    out[counts[bucketOf[i]]++] = i;
  }
  idx.set(out);
}
