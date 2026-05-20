/**
 * HUD: thin wrapper that updates the DOM elements declared in index.html.
 * No framework — the surface is small and `requestAnimationFrame`-driven.
 */

export interface CameraReadout {
  pos: [number, number, number];      // world-space eye position
  target: [number, number, number];   // world-space orbit target
  dir: [number, number, number];      // unit forward vector (eye → target)
  yawRad: number;
  pitchRad: number;
  rollRad: number;
  distance: number;
  fovYRad: number;
  near: number;
  far: number;
}

export class Hud {
  private readonly stateEl = document.getElementById('state')!;
  private readonly formatEl = document.getElementById('format')!;
  private readonly splatsEl = document.getElementById('splats')!;
  private readonly fpsEl = document.getElementById('fps')!;
  private readonly psnrRow = document.getElementById('psnr-row')!;
  private readonly psnrEl = document.getElementById('psnr')!;
  private readonly rnameEl = document.getElementById('rname')!;
  // Camera readout DOM
  private readonly camPosEl = document.getElementById('cam-pos')!;
  private readonly camTargetEl = document.getElementById('cam-target')!;
  private readonly camDirEl = document.getElementById('cam-dir')!;
  private readonly camYawPitchEl = document.getElementById('cam-yawpitch')!;
  private readonly camRollDistEl = document.getElementById('cam-rolldist')!;
  private readonly camFrustumEl = document.getElementById('cam-frustum')!;
  private readonly camCopyBtn = document.getElementById('cam-copy') as HTMLButtonElement | null;
  private lastCam: CameraReadout | null = null;

  constructor() {
    if (this.camCopyBtn) {
      this.camCopyBtn.addEventListener('click', () => {
        if (!this.lastCam) return;
        const json = JSON.stringify(this.lastCam, null, 2);
        void navigator.clipboard.writeText(json).then(() => {
          const orig = this.camCopyBtn!.textContent;
          this.camCopyBtn!.textContent = 'copied ✓';
          setTimeout(() => { this.camCopyBtn!.textContent = orig; }, 900);
        });
      });
    }
  }

  setRenderer(name: string): void { this.rnameEl.textContent = name; }
  setSplats(count: number): void { this.splatsEl.textContent = formatNumber(count); }
  setFormat(s: string): void { this.formatEl.textContent = s; }
  setFps(fps: number): void { this.fpsEl.textContent = fps.toFixed(0); }
  setPsnr(p?: number): void {
    if (p === undefined) { this.psnrRow.style.display = 'none'; return; }
    this.psnrRow.style.display = '';
    this.psnrEl.textContent = `${p.toFixed(2)} dB`;
  }
  setState(s: string, kind?: 'ok' | 'err'): void {
    this.stateEl.textContent = s;
    this.stateEl.classList.remove('ok', 'err');
    if (kind) this.stateEl.classList.add(kind);
  }
  setCamera(c: CameraReadout): void {
    this.lastCam = c;
    this.camPosEl.textContent = vec3(c.pos);
    this.camTargetEl.textContent = vec3(c.target);
    this.camDirEl.textContent = vec3(c.dir);
    this.camYawPitchEl.textContent = `${deg(c.yawRad)}°, ${deg(c.pitchRad)}°`;
    this.camRollDistEl.textContent = `${deg(c.rollRad)}°, ${c.distance.toFixed(2)}`;
    this.camFrustumEl.textContent = `${deg(c.fovYRad)}°, ${c.near.toFixed(3)}, ${c.far.toFixed(0)}`;
  }
}

function formatNumber(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function vec3(v: [number, number, number]): string {
  return `${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)}`;
}

function deg(rad: number): string {
  return ((rad * 180) / Math.PI).toFixed(1);
}
