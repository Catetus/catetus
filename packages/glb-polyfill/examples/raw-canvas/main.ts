/**
 * Minimal WebGL2 viewer that demonstrates the full glb-polyfill decode
 * path. Renders each splat as a single GL_POINT with its DC color and
 * opacity — no Gaussian rasterization, no SH evaluation. The point of this
 * example is to prove that the decode pipeline is everything you need to
 * get from `.glb` bytes to per-splat typed arrays.
 *
 * Drop `scene.glb` (and optional `scene.glb.shpal`, `scene.glb.v5tail`)
 * next to this file and run: `python -m http.server` in this directory.
 */
import {
  decodeSFExtensions,
  decodeV5TailBytes,
  applyV5TailToScene,
} from '@catetus/glb-polyfill';

const SCENE_URL = './scene.glb';
const SHPAL_URL = './scene.glb.shpal'; // optional
const V5TAIL_URL = './scene.glb.v5tail'; // optional

const hud = document.getElementById('hud')!;

/** Split a GLB into its JSON and BIN chunks. */
function splitGlb(bytes: Uint8Array): { json: unknown; bin: Uint8Array } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  const total = dv.getUint32(8, true);
  let off = 12;
  let json: unknown = null;
  let bin = new Uint8Array(0);
  while (off + 8 <= total) {
    const len = dv.getUint32(off, true);
    const type = dv.getUint32(off + 4, true);
    const slice = bytes.subarray(off + 8, off + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(slice));
    else if (type === 0x004e4942) bin = slice;
    off += 8 + len;
  }
  if (!json) throw new Error('GLB missing JSON chunk');
  return { json, bin };
}

async function fetchBytes(url: string): Promise<Uint8Array | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return new Uint8Array(await r.arrayBuffer());
  } catch {
    return null;
  }
}

async function loadScene() {
  hud.textContent = `fetching ${SCENE_URL}…`;
  const glbBytes = await fetchBytes(SCENE_URL);
  if (!glbBytes) throw new Error(`missing ${SCENE_URL}`);
  const { json, bin } = splitGlb(glbBytes);

  const palBytes = await fetchBytes(SHPAL_URL);
  const sidecars = palBytes ? { [SHPAL_URL.replace('./', '')]: palBytes } : undefined;

  hud.textContent = 'decoding…';
  const scene = decodeSFExtensions(json, bin, sidecars);

  // Polyfill returns LINEAR scales + LINEAR opacities (eagerly de-logged /
  // de-logited from SF_log_quant_attrs values). Nothing to do here.
  const scales = scene.scales;
  const opacities = scene.opacities;

  // Optional V5.2 residual sidecar.
  const tailBytes = await fetchBytes(V5TAIL_URL);
  if (tailBytes) {
    const tail = decodeV5TailBytes(tailBytes);
    applyV5TailToScene(
      {
        positions: scene.positions,
        rotations: scene.rotations,
        scales,
        opacities,
        dcRaw: scene.dcRaw,
        shRest: scene.sh_rest,
        shRestCoefs: scene.sh_rest ? scene.sh_rest.length / scene.count / 3 : 0,
      },
      tail,
    );
  }

  return { count: scene.count, positions: scene.positions, dc: scene.dcRaw, opacity: opacities, bbox: scene.bbox };
}

function render(scene: Awaited<ReturnType<typeof loadScene>>) {
  const canvas = document.getElementById('gl') as HTMLCanvasElement;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  const gl = canvas.getContext('webgl2')!;
  gl.clearColor(0.07, 0.07, 0.08, 1);

  // colors = clamp(SH_C0 * dc + 0.5, 0, 1) * opacity-as-alpha.
  const SH_C0 = 0.28209479177387814;
  const colors = new Float32Array(scene.count * 4);
  for (let i = 0; i < scene.count; i++) {
    colors[i * 4 + 0] = Math.min(1, Math.max(0, SH_C0 * scene.dc[i * 3 + 0] + 0.5));
    colors[i * 4 + 1] = Math.min(1, Math.max(0, SH_C0 * scene.dc[i * 3 + 1] + 0.5));
    colors[i * 4 + 2] = Math.min(1, Math.max(0, SH_C0 * scene.dc[i * 3 + 2] + 0.5));
    colors[i * 4 + 3] = scene.opacity[i];
  }

  const vs = `#version 300 es
    layout(location=0) in vec3 a_pos;
    layout(location=1) in vec4 a_col;
    uniform mat4 u_mvp;
    out vec4 v_col;
    void main(){ gl_Position = u_mvp * vec4(a_pos, 1.0); gl_PointSize = 2.0; v_col = a_col; }`;
  const fs = `#version 300 es
    precision mediump float;
    in vec4 v_col; out vec4 o;
    void main(){ o = v_col; }`;
  const prog = link(gl, vs, fs);
  const u_mvp = gl.getUniformLocation(prog, 'u_mvp');

  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  bindBuf(gl, 0, scene.positions, 3);
  bindBuf(gl, 1, colors, 4);

  // Simple orbit camera around the bbox center.
  const c = scene.bbox
    ? [(scene.bbox.min[0]+scene.bbox.max[0])/2, (scene.bbox.min[1]+scene.bbox.max[1])/2, (scene.bbox.min[2]+scene.bbox.max[2])/2]
    : [0,0,0];
  const r = scene.bbox
    ? Math.hypot(scene.bbox.max[0]-scene.bbox.min[0], scene.bbox.max[1]-scene.bbox.min[1], scene.bbox.max[2]-scene.bbox.min[2]) * 0.8
    : 3;

  gl.useProgram(prog);
  gl.enable(gl.DEPTH_TEST);

  function tick(t: number) {
    const a = t * 0.0003;
    const eye = [c[0] + Math.cos(a)*r, c[1] + r*0.4, c[2] + Math.sin(a)*r];
    const mvp = mvpMatrix(eye as [number,number,number], c as [number,number,number], canvas.width / canvas.height);
    gl.uniformMatrix4fv(u_mvp, false, mvp);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.drawArrays(gl.POINTS, 0, scene.count);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function bindBuf(gl: WebGL2RenderingContext, loc: number, data: Float32Array, n: number) {
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const compile = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? '');
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? '');
  return p;
}

function mvpMatrix(eye: [number,number,number], center: [number,number,number], aspect: number): Float32Array {
  // Perspective * lookAt, hand-rolled to keep the example dependency-free.
  const f = 1 / Math.tan(Math.PI / 6);
  const near = 0.05, far = 500;
  const proj = new Float32Array([
    f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)/(near-far),-1, 0,0,(2*far*near)/(near-far),0,
  ]);
  const z = norm3([eye[0]-center[0], eye[1]-center[1], eye[2]-center[2]]);
  const x = norm3(cross([0,1,0], z));
  const y = cross(z, x);
  const view = new Float32Array([
    x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0,
    -dot(x,eye),-dot(y,eye),-dot(z,eye),1,
  ]);
  return mul(proj, view);
}
function cross(a:number[], b:number[]){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function dot(a:number[], b:number[]){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
function norm3(v:number[]){ const l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }
function mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let i=0;i<4;i++) for (let j=0;j<4;j++) for (let k=0;k<4;k++) o[i*4+j] += a[k*4+j] * b[i*4+k];
  return o;
}

loadScene().then((s) => { hud.textContent = `${s.count.toLocaleString()} splats`; render(s); })
  .catch((e) => { hud.textContent = `error: ${e.message}`; console.error(e); });
