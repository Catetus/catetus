/**
 * WebGL2 fallback renderer. Same algorithm as the WebGPU backend: instanced
 * quad per splat, CPU-sorted back-to-front, EWA-style Gaussian fragment
 * shader. Uses `gl.vertexAttribDivisor` to feed per-instance attributes and
 * `gl.drawArraysInstanced` for the draw call.
 */
import type { CameraPose } from '../camera.js';
import type { ChunkDescriptor } from '../manifest.js';
import {
  decodeSplats,
  sortBackToFront,
  type DecodedSplat,
  type Renderer,
  type RendererInitOptions,
} from './base.js';
import {
  buildViewProj,
  computeCovariance3D,
  projectCovariance2D,
  projectPoint,
} from './math.js';

/** Vertex shader. Reads per-instance attributes laid out to match the WebGPU pipeline. */
const VS_SOURCE = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_corner;       // quad corner in [-1,1]
layout(location = 1) in vec4 a_clipPos;      // xy=ndc, z=ndcZ, w=clipW
layout(location = 2) in vec4 a_cov;          // c00, c01, c11, radiusPx
layout(location = 3) in vec4 a_color;        // rgb + opacity
uniform vec2 u_viewportSize;
out vec2 v_offset;
out vec3 v_cov;
out vec4 v_color;
void main() {
  float radiusPx = max(a_cov.w, 1.0);
  vec2 ndcOffset = a_corner * radiusPx * 2.0 / u_viewportSize;
  gl_Position = vec4(
    a_clipPos.x + ndcOffset.x,
    a_clipPos.y + ndcOffset.y,
    clamp(a_clipPos.z, 0.0, 1.0),
    1.0
  );
  v_offset = a_corner * radiusPx;
  v_cov = a_cov.xyz;
  v_color = a_color;
}`;

/** Fragment shader: evaluate the 2D Gaussian and emit premultiplied alpha. */
const FS_SOURCE = `#version 300 es
precision highp float;
in vec2 v_offset;
in vec3 v_cov;
in vec4 v_color;
out vec4 outColor;
void main() {
  float c00 = v_cov.x;
  float c01 = v_cov.y;
  float c11 = v_cov.z;
  float det = max(c00 * c11 - c01 * c01, 1e-6);
  float inv00 =  c11 / det;
  float inv01 = -c01 / det;
  float inv11 =  c00 / det;
  vec2 d = v_offset;
  float power = -0.5 * (d.x * d.x * inv00 + 2.0 * d.x * d.y * inv01 + d.y * d.y * inv11);
  if (power > 0.0) discard;
  float alpha = clamp(v_color.a * exp(power), 0.0, 0.999);
  if (alpha < 1.0 / 255.0) discard;
  outColor = vec4(v_color.rgb * alpha, alpha);
}`;

interface UploadedChunk {
  descriptor: ChunkDescriptor;
  splats: DecodedSplat[];
}

/** Per-instance floats: vec4 clipPos + vec4 cov + vec4 color. */
const FLOATS_PER_INSTANCE = 12;
/** Quad corners drawn as a triangle-strip. */
const VERTICES_PER_QUAD = 4;

/** WebGL2 implementation of {@link Renderer}. */
export class WebGL2Renderer implements Renderer {
  readonly kind = 'webgl2' as const;
  private gl?: WebGL2RenderingContext;
  private program?: WebGLProgram;
  private vao?: WebGLVertexArrayObject;
  private quadBuffer?: WebGLBuffer;
  private instanceBuffer?: WebGLBuffer;
  private instanceCapacity = 0;
  private uViewportSize?: WebGLUniformLocation | null;
  private clear: [number, number, number, number] = [0, 0, 0, 1];
  private chunks: UploadedChunk[] = [];
  /**
   * Number of `drawArraysInstanced` calls the renderer has issued. Exposed
   * for tests so they can assert a frame produced GPU work.
   */
  drawCallCount = 0;

  async init(opts: RendererInitOptions): Promise<void> {
    const gl = opts.canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: true });
    if (!gl) throw new Error('renderer_unavailable: webgl2 unsupported');
    this.gl = gl;
    this.clear = opts.clearColor ?? [0, 0, 0, 1];
    this.program = this.linkProgram(gl, VS_SOURCE, FS_SOURCE);
    this.uViewportSize = gl.getUniformLocation(this.program, 'u_viewportSize');

    // Static quad VBO (4 corners, triangle-strip order).
    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const qb = gl.createBuffer();
    if (!qb) throw new Error('renderer_init_failed: createBuffer (quad)');
    this.quadBuffer = qb;
    gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);

    // Instance VBO created lazily; reserve a placeholder so VAO setup works.
    const ib = gl.createBuffer();
    if (!ib) throw new Error('renderer_init_failed: createBuffer (instance)');
    this.instanceBuffer = ib;

    // VAO binds both buffers + divisors.
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('renderer_init_failed: createVertexArray');
    this.vao = vao;
    gl.bindVertexArray(vao);

    // location 0 — quad corner (per-vertex).
    gl.bindBuffer(gl.ARRAY_BUFFER, qb);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(0, 0);

    // locations 1..3 — per-instance.
    gl.bindBuffer(gl.ARRAY_BUFFER, ib);
    const stride = FLOATS_PER_INSTANCE * 4;
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, stride, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 16);
    gl.vertexAttribDivisor(2, 1);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, stride, 32);
    gl.vertexAttribDivisor(3, 1);

    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(this.clear[0], this.clear[1], this.clear[2], this.clear[3]);
  }

  uploadChunk(descriptor: ChunkDescriptor, bytes: Uint8Array): void {
    if (!this.gl) throw new Error('renderer_init_failed: not initialized');
    const splats = decodeSplats(bytes);
    this.chunks.push({ descriptor, splats });
  }

  async renderFrame(camera: CameraPose): Promise<void> {
    const gl = this.gl;
    if (!gl || !this.program || !this.vao || !this.instanceBuffer) {
      throw new Error('renderer_init_failed: not initialized');
    }
    const width = Math.max(gl.drawingBufferWidth, 1);
    const height = Math.max(gl.drawingBufferHeight, 1);
    const aspect = width / height;
    const { view, viewProj } = buildViewProj(camera, aspect);
    const focalY = height / (2 * Math.tan(camera.fovY * 0.5));
    const focalX = focalY;

    const all = this.flattenSplats();
    const count = all.length;
    const indices = new Uint32Array(count);
    for (let i = 0; i < count; i++) indices[i] = i;
    sortBackToFront(all, camera, indices);

    const data = new Float32Array(count * FLOATS_PER_INSTANCE);
    for (let i = 0; i < count; i++) {
      const s = all[indices[i]!]!;
      const proj = projectPoint(s.position, viewProj);
      const vz = view[2]! * s.position[0] + view[6]! * s.position[1] + view[10]! * s.position[2] + view[14]!;
      const depth = -vz;
      const behind = depth <= 0;
      const cov3 = computeCovariance3D(s.scale, s.rotation);
      const [c00, c01, c11] = behind
        ? [1, 0, 1]
        : projectCovariance2D(cov3, view, focalX, focalY, depth);
      const trace = c00 + c11;
      const halfTrace = trace * 0.5;
      const term = Math.sqrt(Math.max(halfTrace * halfTrace - (c00 * c11 - c01 * c01), 0));
      const lambdaMax = halfTrace + term;
      const radius = behind ? 0 : 3 * Math.sqrt(Math.max(lambdaMax, 0));
      const o = i * FLOATS_PER_INSTANCE;
      data[o + 0] = proj.ndc[0];
      data[o + 1] = proj.ndc[1];
      data[o + 2] = proj.ndc[2];
      data[o + 3] = proj.w;
      data[o + 4] = c00;
      data[o + 5] = c01;
      data[o + 6] = c11;
      data[o + 7] = radius;
      data[o + 8] = s.colorDC[0];
      data[o + 9] = s.colorDC[1];
      data[o + 10] = s.colorDC[2];
      data[o + 11] = s.opacity;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
    if (count > this.instanceCapacity) {
      const cap = Math.max(count, Math.ceil(count * 1.5));
      gl.bufferData(gl.ARRAY_BUFFER, cap * FLOATS_PER_INSTANCE * 4, gl.DYNAMIC_DRAW);
      this.instanceCapacity = cap;
    }
    if (count > 0) {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    if (this.uViewportSize) gl.uniform2f(this.uViewportSize, width, height);
    gl.bindVertexArray(this.vao);
    if (count > 0) {
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, VERTICES_PER_QUAD, count);
      this.drawCallCount++;
    }
    gl.bindVertexArray(null);
  }

  async readPixels(): Promise<Uint8Array> {
    const gl = this.gl;
    if (!gl) throw new Error('renderer_init_failed: not initialized');
    const w = gl.drawingBufferWidth;
    const h = gl.drawingBufferHeight;
    const buf = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    return buf;
  }

  destroy(): void {
    const gl = this.gl;
    if (gl) {
      if (this.program) gl.deleteProgram(this.program);
      if (this.vao) gl.deleteVertexArray(this.vao);
      if (this.quadBuffer) gl.deleteBuffer(this.quadBuffer);
      if (this.instanceBuffer) gl.deleteBuffer(this.instanceBuffer);
    }
    this.chunks = [];
    this.gl = undefined;
    this.program = undefined;
    this.vao = undefined;
    this.quadBuffer = undefined;
    this.instanceBuffer = undefined;
    this.instanceCapacity = 0;
  }

  private flattenSplats(): DecodedSplat[] {
    let total = 0;
    for (const c of this.chunks) total += c.splats.length;
    const out: DecodedSplat[] = new Array(total);
    let w = 0;
    for (const c of this.chunks) {
      for (const s of c.splats) out[w++] = s;
    }
    return out;
  }

  private linkProgram(gl: WebGL2RenderingContext, vs: string, fs: string): WebGLProgram {
    const compile = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type);
      if (!sh) throw new Error('renderer_init_failed: createShader');
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh) ?? 'unknown';
        gl.deleteShader(sh);
        throw new Error(`renderer_init_failed: shader compile: ${log}`);
      }
      return sh;
    };
    const v = compile(gl.VERTEX_SHADER, vs);
    const f = compile(gl.FRAGMENT_SHADER, fs);
    const program = gl.createProgram();
    if (!program) throw new Error('renderer_init_failed: createProgram');
    gl.attachShader(program, v);
    gl.attachShader(program, f);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) ?? 'unknown';
      gl.deleteProgram(program);
      throw new Error(`renderer_init_failed: program link: ${log}`);
    }
    gl.deleteShader(v);
    gl.deleteShader(f);
    return program;
  }
}
