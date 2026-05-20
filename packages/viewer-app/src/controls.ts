/**
 * Mouse + touch + keyboard orbit controls.
 *
 * Mouse:
 *   - left-drag = orbit
 *   - right-drag (or shift+drag) = pan (in screen-space, mapped through
 *     `distance * tan(fov)` so the drag tracks the cursor under the scene).
 *   - wheel = dolly (zoom)
 *
 * Touch:
 *   - 1 finger = orbit
 *   - 2 fingers panning = pan; 2 fingers pinching = dolly.
 *
 * Keyboard:
 *   - WASD = pan along view-right / view-forward (S/W push toward target).
 *   - Q/E  = roll left/right
 *   - R    = reset to the camera's "home" pose
 *   - F    = frame the supplied bbox provider (set via setFrameAll).
 *
 * Damping: every drag/scroll updates a target state; render loop lerps the
 * live `cam` toward the target with `1 - exp(-dt/tau)`. This gives the
 * "inertia, no snap-jumps" feel without any external dep.
 */
import type { CameraState } from './camera.js';
import { frameCameraToBbox } from './camera.js';

interface Vec2 { x: number; y: number; }

export interface ControlsOptions {
  canvas: HTMLCanvasElement;
  cam: CameraState;
  getFrameBbox?: () => { min: [number, number, number]; max: [number, number, number] } | null;
  /** Returns flat XYZ positions for ray-picking the orbit pivot on double-click. */
  getScenePositions?: () => Float32Array | null;
}

export class OrbitControls {
  private readonly canvas: HTMLCanvasElement;
  /** Live (rendered) camera. */
  readonly cam: CameraState;
  /** Target camera state; drifted into `cam` per-frame for damping. */
  private readonly target: CameraState;
  /** Pristine pose used by `reset()`. */
  private home: CameraState;
  private getFrameBbox?: () => { min: [number, number, number]; max: [number, number, number] } | null;
  private getScenePositions?: () => Float32Array | null;

  private dragging = false;
  private panning = false;
  /** FPS mouse-look: rotates look direction in place, eye stays fixed. */
  private freelook = false;
  private lastPos: Vec2 = { x: 0, y: 0 };
  /** Active touches keyed by pointerId. */
  private touches = new Map<number, Vec2>();
  private prevPinch = 0;
  /** Set of currently-held keys. */
  private keys = new Set<string>();

  constructor(opts: ControlsOptions) {
    this.canvas = opts.canvas;
    this.cam = opts.cam;
    this.target = cloneCam(opts.cam);
    this.home = cloneCam(opts.cam);
    this.getFrameBbox = opts.getFrameBbox;
    this.getScenePositions = opts.getScenePositions;
    this.attach();
  }

  setScenePositionsProvider(fn: () => Float32Array | null): void {
    this.getScenePositions = fn;
  }

  /** Update home pose (call after framing the initial scene). */
  setHome(cam: CameraState): void {
    this.home = cloneCam(cam);
    Object.assign(this.target, cloneCam(cam));
  }

  setFrameBboxProvider(
    fn: () => { min: [number, number, number]; max: [number, number, number] } | null,
  ): void {
    this.getFrameBbox = fn;
  }

  /** Tween `cam` toward `target` and apply per-frame keyboard movement. */
  update(dtMs: number): void {
    this.applyKeyboard(dtMs);
    const tau = 90; // ms
    const a = 1 - Math.exp(-dtMs / tau);
    this.cam.yaw     = lerp(this.cam.yaw,     this.target.yaw,     a);
    this.cam.pitch   = lerp(this.cam.pitch,   this.target.pitch,   a);
    this.cam.roll    = lerp(this.cam.roll,    this.target.roll,    a);
    this.cam.distance = lerp(this.cam.distance, this.target.distance, a);
    this.cam.target[0] = lerp(this.cam.target[0], this.target.target[0], a);
    this.cam.target[1] = lerp(this.cam.target[1], this.target.target[1], a);
    this.cam.target[2] = lerp(this.cam.target[2], this.target.target[2], a);
  }

  reset(): void {
    Object.assign(this.target, cloneCam(this.home));
  }

  frameAll(): void {
    const bbox = this.getFrameBbox?.();
    if (!bbox) return;
    frameCameraToBbox(this.target, bbox);
    this.home = cloneCam(this.target);
  }

  /* ------------------------------------------------------------ */
  /* Event wiring                                                 */
  /* ------------------------------------------------------------ */

  private attach(): void {
    const c = this.canvas;
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('pointerdown', this.onPointerDown);
    c.addEventListener('pointermove', this.onPointerMove);
    c.addEventListener('pointerup', this.onPointerUp);
    c.addEventListener('pointercancel', this.onPointerUp);
    c.addEventListener('lostpointercapture', this.onPointerUp);
    c.addEventListener('wheel', this.onWheel, { passive: false });
    c.addEventListener('dblclick', this.onDoubleClick);
    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);
  }

  /**
   * Mouse button conventions (mirrors Unreal Editor's split, with trackpad
   * fallbacks since Mac trackpads make right/middle click awkward):
   *   - Left button (no modifier)  → orbit around current pivot
   *   - Right button               → free-look (FPS mouse-look, eye fixed)
   *   - Middle button              → pan
   *   - Shift + Left (trackpad)    → pan
   *   - Alt   + Left (trackpad)    → free-look
   * Modifier checks happen in priority order: Alt > Shift > base.
   */
  private readonly onPointerDown = (e: PointerEvent): void => {
    this.canvas.setPointerCapture(e.pointerId);
    if (e.pointerType === 'touch') {
      this.touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
      this.prevPinch = pinchDist(this.touches);
      return;
    }
    const left = e.button === 0;
    const right = e.button === 2;
    const middle = e.button === 1;
    // Trackpad-friendly modifier fallbacks (priority: Alt > Shift > base).
    this.freelook = right || (left && e.altKey);
    this.panning  = !this.freelook && (middle || (left && e.shiftKey));
    this.dragging = !this.freelook && !this.panning && left;
    this.lastPos = { x: e.clientX, y: e.clientY };
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (e.pointerType === 'touch') return this.handleTouchMove(e);
    if (!this.dragging && !this.panning && !this.freelook) return;
    const dx = e.clientX - this.lastPos.x;
    const dy = e.clientY - this.lastPos.y;
    this.lastPos = { x: e.clientX, y: e.clientY };
    if (this.freelook)       this.mouseLook(dx, dy);
    else if (this.panning)   this.pan(dx, dy);
    else                     this.orbit(dx, dy);
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    try { this.canvas.releasePointerCapture(e.pointerId); } catch { /* fine */ }
    if (e.pointerType === 'touch') {
      this.touches.delete(e.pointerId);
      this.prevPinch = pinchDist(this.touches);
      return;
    }
    this.dragging = false;
    this.panning = false;
    this.freelook = false;
  };

  /**
   * Double-click LMB: ray-cast against splat positions, set that point as the
   * new orbit pivot. The eye stays put; yaw/pitch/distance recompute so the
   * camera is now facing the picked splat while orbiting around it on the
   * next drag.
   */
  private readonly onDoubleClick = (e: MouseEvent): void => {
    if (e.button !== 0) return;
    const rect = this.canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    this.pickAndRetarget(px, py, rect.width, rect.height);
  };

  /**
   * Free-look: drag pixels → look-direction delta. Convert to yaw/pitch
   * radians using a fixed sensitivity and apply via lookInPlace so the eye
   * stays fixed (true FPS turn-in-place).
   */
  private mouseLook(dxPx: number, dyPx: number): void {
    const sens = 0.0035; // rad per pixel
    this.lookInPlace(dxPx * sens, dyPx * sens);
  }

  private handleTouchMove(e: PointerEvent): void {
    const prev = this.touches.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    prev.x = e.clientX; prev.y = e.clientY;
    if (this.touches.size === 1) {
      this.orbit(dx, dy);
    } else if (this.touches.size >= 2) {
      // Average pan from all moving fingers.
      this.pan(dx, dy);
      const pinch = pinchDist(this.touches);
      if (this.prevPinch > 0 && pinch > 0) {
        const delta = pinch - this.prevPinch;
        // pinching out = closer (smaller distance).
        this.dolly(-delta * 6);
      }
      this.prevPinch = pinch;
    }
  }

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    // Normalize lines/px/page to roughly "px-per-tick".
    const px = e.deltaMode === 0 ? e.deltaY : e.deltaY * 33;
    this.dolly(px);
  };

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    const k = e.key.toLowerCase();
    this.keys.add(k);
    if (k === 'r') { this.reset(); e.preventDefault(); }
    if (k === 'f') { this.frameAll(); e.preventDefault(); }
  };
  private readonly onKeyUp = (e: KeyboardEvent): void => {
    this.keys.delete(e.key.toLowerCase());
  };

  /* ------------------------------------------------------------ */
  /* Manipulators                                                 */
  /* ------------------------------------------------------------ */

  private orbit(dxPx: number, dyPx: number): void {
    const speed = 0.005;
    this.target.yaw   -= dxPx * speed;
    // Pitch is UNBOUNDED — camera can flip over the top continuously. The view
    // matrix's up vector is derived from sign(cos(pitch)) below in camera.ts so
    // there's no degenerate flip even when the user spins past the poles.
    this.target.pitch += dyPx * speed;
  }

  private pan(dxPx: number, dyPx: number): void {
    const cam = this.target;
    // World units per pixel at the target plane:
    //   half-screen-world = distance * tan(fov/2)
    //   ⇒ units/px = 2 * half-screen-world / canvasHeight
    const h = Math.max(1, this.canvas.clientHeight);
    const panScale = (cam.distance * Math.tan(cam.fovYRad * 0.5) * 2) / h;
    // Eye direction from target: (cp*sy, sp, cp*cy). Forward = -that.
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw),    sy = Math.sin(cam.yaw);
    const fx = -cp * sy, fy = -sp, fz = -cp * cy;
    // right = cross(forward, world-up=(0,1,0)) = (fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0)
    //       = (-fz, 0, fx)
    let rx = -fz, ry = 0, rz = fx;
    const rlen = Math.hypot(rx, ry, rz) || 1;
    rx /= rlen; ry /= rlen; rz /= rlen;
    // up = cross(right, forward), already unit.
    const ux = ry * fz - rz * fy;
    const uy = rz * fx - rx * fz;
    const uz = rx * fy - ry * fx;
    // Drag right ⇒ scene shifts right under the cursor ⇒ target moves left.
    const sX = -dxPx * panScale;
    const sY =  dyPx * panScale;
    cam.target[0] += rx * sX + ux * sY;
    cam.target[1] += ry * sX + uy * sY;
    cam.target[2] += rz * sX + uz * sY;
  }

  private dolly(deltaPx: number): void {
    const factor = Math.exp(deltaPx * 0.0015);
    this.target.distance = clamp(this.target.distance * factor, 0.01, 1e6);
  }

  /**
   * Ray-cast against the splat positions to find what the user clicked,
   * reparent the orbit pivot to that point, and rebuild (yaw, pitch, distance)
   * to keep the eye fixed. Invoked ONLY on explicit double-click — never on
   * every drag — so it can't surprise users mid-orbit.
   *
   * Algorithm: O(N) "closest splat to the click ray in angular space."
   * ~10-15 ms on 1.2M splats; no acceleration structure required.
   */
  private pickAndRetarget(canvasPx: number, canvasPy: number, w: number, h: number): void {
    const positions = this.getScenePositions?.();
    if (!positions) return;

    const ndcX = (canvasPx / w) * 2 - 1;
    const ndcY = 1 - (canvasPy / h) * 2;

    const cam = this.cam;
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cyaw = Math.cos(cam.yaw), syaw = Math.sin(cam.yaw);
    const ex = cam.target[0] + cam.distance * cp * syaw;
    const ey = cam.target[1] + cam.distance * sp;
    const ez = cam.target[2] + cam.distance * cp * cyaw;
    const fx = -cp * syaw, fy = -sp, fz = -cp * cyaw;
    let rx = -fz, ry = 0, rz = fx;
    const rlen = Math.hypot(rx, ry, rz) || 1;
    rx /= rlen; ry /= rlen; rz /= rlen;
    let ux = ry * fz - rz * fy;
    let uy = rz * fx - rx * fz;
    let uz = rx * fy - ry * fx;
    if (cam.roll !== 0) {
      const cr = Math.cos(cam.roll), sr = Math.sin(cam.roll);
      const rx2 =  cr * rx + sr * ux;
      const ry2 =  cr * ry + sr * uy;
      const rz2 =  cr * rz + sr * uz;
      const ux2 = -sr * rx + cr * ux;
      const uy2 = -sr * ry + cr * uy;
      const uz2 = -sr * rz + cr * uz;
      rx = rx2; ry = ry2; rz = rz2;
      ux = ux2; uy = uy2; uz = uz2;
    }
    const tanHalfFov = Math.tan(cam.fovYRad * 0.5);
    const aspect = w / Math.max(h, 1);
    const sx = ndcX * tanHalfFov * aspect;
    const sy = ndcY * tanHalfFov;
    let dirX = fx + rx * sx + ux * sy;
    let dirY = fy + ry * sx + uy * sy;
    let dirZ = fz + rz * sx + uz * sy;
    const dl = Math.hypot(dirX, dirY, dirZ) || 1;
    dirX /= dl; dirY /= dl; dirZ /= dl;

    const N = positions.length / 3;
    let bestScore = Infinity;
    let bestIdx = -1;
    for (let i = 0; i < N; i++) {
      const px = positions[i * 3 + 0] - ex;
      const py = positions[i * 3 + 1] - ey;
      const pz = positions[i * 3 + 2] - ez;
      const along = px * dirX + py * dirY + pz * dirZ;
      if (along <= 0) continue;
      const sq = px * px + py * py + pz * pz - along * along;
      if (sq < 0) continue;
      const score = sq / (along * along);
      if (score < bestScore) { bestScore = score; bestIdx = i; }
    }
    if (bestIdx < 0) return;

    const tx = positions[bestIdx * 3 + 0];
    const ty = positions[bestIdx * 3 + 1];
    const tz = positions[bestIdx * 3 + 2];
    const distance = Math.hypot(tx - ex, ty - ey, tz - ez);
    if (distance < 1e-5) return;
    const newPitch = Math.asin((ey - ty) / distance);
    const newYaw   = Math.atan2(ex - tx, ez - tz);
    this.cam.target = [tx, ty, tz];
    this.cam.yaw = newYaw;
    this.cam.pitch = newPitch;
    this.cam.distance = distance;
    this.target.target = [tx, ty, tz];
    this.target.yaw = newYaw;
    this.target.pitch = newPitch;
    this.target.distance = distance;
  }

  /**
   * Rotate look direction by (dYaw, dPitch) while keeping the eye position
   * FIXED. We achieve this by recomputing the target so that
   *   target = eye + distance * dir(newYaw, newPitch)
   * (with `dir` pointing from eye toward target). This is the FPS "turn your
   * head" gesture rather than orbital "spin around the model."
   */
  private lookInPlace(dYaw: number, dPitch: number): void {
    const cam = this.target;
    // Capture current eye in world space BEFORE mutating yaw/pitch.
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw),    sy = Math.sin(cam.yaw);
    const ex = cam.target[0] + cam.distance * cp * sy;
    const ey = cam.target[1] + cam.distance * sp;
    const ez = cam.target[2] + cam.distance * cp * cy;
    // Apply rotation deltas.
    cam.yaw   += dYaw;
    cam.pitch += dPitch;
    // Recompute target so eye stays put.
    const ncp = Math.cos(cam.pitch), nsp = Math.sin(cam.pitch);
    const ncy = Math.cos(cam.yaw),    nsy = Math.sin(cam.yaw);
    cam.target[0] = ex - cam.distance * ncp * nsy;
    cam.target[1] = ey - cam.distance * nsp;
    cam.target[2] = ez - cam.distance * ncp * ncy;
  }

  private applyKeyboard(dtMs: number): void {
    if (this.keys.size === 0) return;
    const dt = dtMs / 1000;

    // Roll (1/3).
    if (this.keys.has('1')) this.target.roll += 1.5 * dt;
    if (this.keys.has('3')) this.target.roll -= 1.5 * dt;

    // Free-look yaw (Z/C) and pitch (arrowup/arrowdown): rotate look direction
    // while keeping the eye position fixed.
    const lookRate = 1.5; // rad / sec
    let dYaw = 0, dPitch = 0;
    if (this.keys.has('z')) dYaw -= lookRate * dt;
    if (this.keys.has('c')) dYaw += lookRate * dt;
    if (this.keys.has('arrowup'))   dPitch -= lookRate * dt;
    if (this.keys.has('arrowdown')) dPitch += lookRate * dt;
    if (dYaw !== 0 || dPitch !== 0) this.lookInPlace(dYaw, dPitch);

    // World-up vertical translation (Q/E) — the "elevator" third axis.
    let vY = 0;
    if (this.keys.has('e')) vY += 1;
    if (this.keys.has('q')) vY -= 1;

    let pX = 0, pZ = 0;
    if (this.keys.has('w')) pZ -= 1; // forward (along view direction)
    if (this.keys.has('s')) pZ += 1; // back
    if (this.keys.has('a')) pX += 1; // strafe LEFT
    if (this.keys.has('d')) pX -= 1; // strafe RIGHT
    if (pX === 0 && pZ === 0 && vY === 0) return;

    // FPS-style fly: translate BOTH eye and target together along view-right
    // and view-forward, so W keeps moving you through the scene without
    // hitting the dolly-distance floor. Speed tied to current scale.
    const cam = this.target;
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const cy = Math.cos(cam.yaw),    sy = Math.sin(cam.yaw);
    // Forward = -(eye - target normalized) = direction the camera is looking.
    const fx = -cp * sy, fy = -sp, fz = -cp * cy;
    // Right = cross(forward, world-up=(0,1,0))
    let rx = -fz, ry = 0, rz = fx;
    const rlen = Math.hypot(rx, ry, rz) || 1;
    rx /= rlen; ry /= rlen; rz /= rlen;
    if (cam.roll !== 0) {
      const ux0 = ry * fz - rz * fy;
      const uy0 = rz * fx - rx * fz;
      const uz0 = rx * fy - ry * fx;
      const cr = Math.cos(cam.roll), sr = Math.sin(cam.roll);
      rx = cr * rx + sr * ux0;
      ry = cr * ry + sr * uy0;
      rz = cr * rz + sr * uz0;
    }

    // Constant speed tied ONLY to the scene's bbox diagonal — not zoom level.
    // Cross the full scene in ~3 seconds; same speed whether you're close or
    // far. Decoupled from orbit-distance so behavior is predictable.
    const bbox = this.getFrameBbox?.();
    let sceneScale = 1;
    if (bbox) {
      const dx = bbox.max[0] - bbox.min[0];
      const dy = bbox.max[1] - bbox.min[1];
      const dz = bbox.max[2] - bbox.min[2];
      sceneScale = Math.hypot(dx, dy, dz);
    }
    const speed = sceneScale * 0.33 * dt;        // ~3 sec to cross scene
    const tx = rx * pX * speed + (-fx) * pZ * speed;
    const ty = ry * pX * speed + (-fy) * pZ * speed + vY * speed;  // Q/E adds world-Y
    const tz = rz * pX * speed + (-fz) * pZ * speed;
    cam.target[0] += tx;
    cam.target[1] += ty;
    cam.target[2] += tz;
  }
}

function pinchDist(touches: Map<number, Vec2>): number {
  if (touches.size < 2) return 0;
  const arr = Array.from(touches.values());
  const dx = arr[0].x - arr[1].x;
  const dy = arr[0].y - arr[1].y;
  return Math.hypot(dx, dy);
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
function cloneCam(c: CameraState): CameraState {
  return {
    target: [c.target[0], c.target[1], c.target[2]],
    distance: c.distance,
    yaw: c.yaw,
    pitch: c.pitch,
    roll: c.roll,
    fovYRad: c.fovYRad,
    near: c.near,
    far: c.far,
  };
}
