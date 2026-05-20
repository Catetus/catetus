/**
 * Viewer-app entrypoint.
 *
 * Wires up:
 *   - the WebGL2 splat renderer (renderer.ts)
 *   - orbit + WASD + touch controls (controls.ts)
 *   - drag-and-drop / file-picker / `?src=` URL loader (loaders/*)
 *   - HUD (ui/hud.ts)
 *
 * Loading semantics:
 *   - File drop / picker: read every file in the drop, hand to dispatcher.
 *   - `?src=<url>`: fetch bytes, dispatcher auto-fetches the .shpal sibling for
 *     SF GLBs (uri resolved relative to the src URL).
 *
 * Render loop: a single rAF loop ticks controls.update(dt) then draws. We
 * intentionally redraw every frame so the inertia-damped camera shows smooth
 * motion; cost is dominated by re-uploading sorted vertex buffers, which is
 * acceptable up to ~1.5 M splats on a discrete GPU.
 */
import { defaultCamera, viewProjMatrix } from './camera.js';
import { OrbitControls } from './controls.js';
import { SplatRenderer } from './renderer.js';
import { loadFromFiles, loadFromUrl, type NamedBytes } from './loaders/dispatch.js';
import { Hud } from './ui/hud.js';
import { type SplatScene } from './splat-scene.js';

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const dropOverlay = document.getElementById('drop-overlay')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const resetBtn = document.getElementById('reset-cam') as HTMLButtonElement;
const frameBtn = document.getElementById('frame-all') as HTMLButtonElement;
const presetBtn = document.getElementById('goto-preset') as HTMLButtonElement | null;

const hud = new Hud();
hud.setState('idle');
hud.setRenderer('webgl2');

const renderer = new SplatRenderer();
try {
  renderer.init(canvas);
} catch (e) {
  hud.setState(`renderer: ${(e as Error).message}`, 'err');
  throw e;
}

const camera = defaultCamera();
const controls = new OrbitControls({ canvas, cam: camera });

let scene: SplatScene | null = null;

/**
 * Detect mobile UA. Used to gate memory pressure + DPR cap.
 * Heuristic only — `?desktop` query param overrides for testing.
 */
const IS_MOBILE = (() => {
  if (new URLSearchParams(window.location.search).has('desktop')) return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
})();

function resize(): void {
  // Mobile: clamp DPR more aggressively when the active scene is large.
  // The bbox-quad fragment shader cost is quadratic in DPR; an iPhone with
  // dpr=3 on a 1M-splat scene fills the screen with overdraw and OOMs the tab.
  const N = scene?.count ?? 0;
  const baseCap = IS_MOBILE ? (N > 200_000 ? 1.25 : 1.5) : 2;
  const dpr = Math.min(window.devicePixelRatio || 1, baseCap);
  const w = Math.max(1, Math.floor(window.innerWidth * dpr));
  const h = Math.max(1, Math.floor(window.innerHeight * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
}
window.addEventListener('resize', resize);
resize();

/* ----------------------------------------------------------------- */
/* Render loop                                                       */
/* ----------------------------------------------------------------- */

let last = performance.now();
let fpsAccum = 0;
let fpsFrames = 0;
let fpsTimer = 0;
function frame(now: number): void {
  const dt = Math.min(now - last, 100);
  last = now;
  controls.update(dt);
  renderer.render(camera, canvas.width, canvas.height);

  // FPS over a sliding ~500 ms window.
  fpsAccum += dt;
  fpsFrames += 1;
  fpsTimer += dt;
  if (fpsTimer >= 500) {
    const fps = (fpsFrames * 1000) / fpsAccum;
    hud.setFps(fps);
    fpsAccum = 0; fpsFrames = 0; fpsTimer = 0;
  }

  // Camera readout: throttled to ~10 Hz so the HUD doesn't repaint every frame.
  camThrottle += dt;
  if (camThrottle >= 100) {
    camThrottle = 0;
    const aspect = canvas.width / Math.max(canvas.height, 1);
    const vp = viewProjMatrix(camera, aspect);
    const tx = camera.target[0], ty = camera.target[1], tz = camera.target[2];
    let dx = tx - vp.eye[0], dy = ty - vp.eye[1], dz = tz - vp.eye[2];
    const dl = Math.hypot(dx, dy, dz) || 1;
    dx /= dl; dy /= dl; dz /= dl;
    hud.setCamera({
      pos: vp.eye,
      target: camera.target,
      dir: [dx, dy, dz],
      yawRad: camera.yaw,
      pitchRad: camera.pitch,
      rollRad: camera.roll,
      distance: camera.distance,
      fovYRad: camera.fovYRad,
      near: camera.near,
      far: camera.far,
    });
  }

  requestAnimationFrame(frame);
}
let camThrottle = 0;
requestAnimationFrame(frame);

/* ----------------------------------------------------------------- */
/* Loading                                                           */
/* ----------------------------------------------------------------- */

/**
 * User-confirmed inside-the-bonsai-scene preset. Applied by the "Preset view"
 * button only — never automatically — so it doesn't surprise users loading
 * other scenes. Tweak by hitting "Copy camera JSON" then editing this block.
 */
const PRESET_VIEW = {
  target: [45.009692362719996, -24.41910971646988, 95.39478152255184] as [number, number, number],
  yaw: 0.4436210937499999,
  pitch: 2.8957812500000006,
  roll: 0,
  distance: 110.27689233804765,
  fovYRad: 0.8726646259971648,
  near: 0.11027689233804765,
  far: 110276.89233804765,
};

function applyPresetView(cam: import('./camera.js').CameraState): void {
  cam.target = [...PRESET_VIEW.target] as [number, number, number];
  cam.yaw = PRESET_VIEW.yaw;
  cam.pitch = PRESET_VIEW.pitch;
  cam.roll = PRESET_VIEW.roll;
  cam.distance = PRESET_VIEW.distance;
  cam.fovYRad = PRESET_VIEW.fovYRad;
  cam.near = PRESET_VIEW.near;
  cam.far = PRESET_VIEW.far;
}

async function applyScene(s: SplatScene): Promise<void> {
  const isFirstScene = scene === null;
  scene = s;
  // Mobile memory guard: 1M+ splats × ~280 bytes unpacked + texture overhead
  // can OOM iOS tabs (~1 GB kill threshold). Refuse > 500k on mobile, warn 250k-500k.
  if (IS_MOBILE && s.count > 500_000) {
    hud.setState(`refused: scene is ${(s.count/1e6).toFixed(2)}M splats (mobile cap 500k)`, 'err');
    scene = null;
    return;
  }
  if (IS_MOBILE && s.count > 250_000) {
    hud.setState(`warning: ${(s.count/1e6).toFixed(2)}M splats may be slow on mobile`, 'err');
  }
  renderer.setScene(s);
  resize();  // re-evaluate DPR cap with new scene size
  hud.setSplats(s.count);
  hud.setFormat(s.meta.format);
  hud.setPsnr(s.meta.psnr);
  // First scene: auto-frame to bbox and lock that as the home pose so 'R'
  // returns here. Subsequent scenes: KEEP THE CURRENT CAMERA STATE so
  // visual A/B compare works without re-navigating. User can hit F to re-fit
  // or click "Preset view" to jump to the saved bonsai viewpoint.
  if (isFirstScene) {
    // Default to the saved Preset view — empirically lands in a sensible
    // viewpoint for Mip-NeRF 360-style scenes (which all share COLMAP world
    // coordinates). User can hit F to bbox-fit the current scene instead.
    applyPresetView(camera);
    controls.setHome(camera);
  }
  controls.setFrameBboxProvider(() => scene?.bbox ?? null);
  controls.setScenePositionsProvider(() => scene?.positions ?? null);
  hud.setState(`loaded ${s.meta.source}`, 'ok');
}

async function readFilesToBytes(files: FileList | File[]): Promise<NamedBytes[]> {
  const arr = Array.from(files);
  return Promise.all(arr.map(async (f) => ({
    name: f.name,
    bytes: new Uint8Array(await f.arrayBuffer()),
  })));
}

async function loadFiles(files: FileList | File[]): Promise<void> {
  hud.setState(`loading ${files.length} file(s)…`);
  try {
    const bag = await readFilesToBytes(files);
    const s = await loadFromFiles(bag);
    await applyScene(s);
  } catch (e) {
    hud.setState((e as Error).message, 'err');
    console.error(e);
  }
}

async function loadUrl(url: string): Promise<void> {
  hud.setState(`fetching ${url}…`);
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url}: HTTP ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    const s = await loadFromUrl(url, bytes);
    await applyScene(s);
  } catch (e) {
    hud.setState((e as Error).message, 'err');
    console.error(e);
  }
}

/* ----------------------------------------------------------------- */
/* Drag-drop                                                         */
/* ----------------------------------------------------------------- */

let dragCounter = 0;
window.addEventListener('dragenter', (e) => {
  e.preventDefault();
  dragCounter += 1;
  dropOverlay.classList.add('active');
});
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('dragleave', () => {
  dragCounter -= 1;
  if (dragCounter <= 0) { dragCounter = 0; dropOverlay.classList.remove('active'); }
});
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragCounter = 0;
  dropOverlay.classList.remove('active');
  if (!e.dataTransfer) return;
  const files = e.dataTransfer.files;
  if (files.length === 0) return;
  await loadFiles(files);
});

/* ----------------------------------------------------------------- */
/* File picker + buttons                                             */
/* ----------------------------------------------------------------- */

fileInput.addEventListener('change', async () => {
  if (!fileInput.files) return;
  await loadFiles(fileInput.files);
  fileInput.value = '';
});
resetBtn.addEventListener('click', () => controls.reset());
frameBtn.addEventListener('click', () => controls.frameAll());
if (presetBtn) {
  presetBtn.addEventListener('click', () => {
    applyPresetView(camera);
    controls.setHome(camera);
  });
}

// H to toggle SH-rest (view-dependent shading). Off → ~3x faster on dense
// scenes; trades view-dependent specular highlights for the baked DC color.
window.addEventListener('keydown', (e) => {
  if (e.key === 'h' || e.key === 'H') {
    if (e.target instanceof HTMLInputElement) return;  // don't fire while typing
    const next = !renderer.isShRestEnabled();
    renderer.setShRestEnabled(next);
    hud.setState(`SH-rest ${next ? 'ON' : 'OFF (fast mode)'}`, next ? 'ok' : 'ok');
  }
});

/* ----------------------------------------------------------------- */
/* URL param loading                                                 */
/* ----------------------------------------------------------------- */

const params = new URLSearchParams(window.location.search);
const srcParam = params.get('src');
if (srcParam) {
  void loadUrl(srcParam);
}
