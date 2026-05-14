/**
 * Opt-in on-canvas FPS / memory HUD. Render-thread cost is one DOM textContent
 * write per frame; skipped entirely when `stats: false`.
 */
/**
 * Tiny FPS / memory overlay.
 *
 * Memory only renders when `performance.memory` is exposed (Chromium). Other
 * browsers see just the FPS line.
 */
export class StatsOverlay {
    el;
    samples = [];
    windowMs;
    lastTick = 0;
    constructor(opts) {
        this.windowMs = opts.windowMs ?? 1000;
        this.el = document.createElement('div');
        this.el.dataset.role = 'splatforge-stats';
        Object.assign(this.el.style, {
            position: 'absolute',
            top: '8px',
            left: '8px',
            padding: '4px 6px',
            font: '11px/1.2 ui-monospace, monospace',
            color: '#0f0',
            background: 'rgba(0,0,0,0.6)',
            pointerEvents: 'none',
            zIndex: '1000',
        });
        opts.anchor.appendChild(this.el);
    }
    /** Mark a frame boundary. Call once per `renderFrame`. */
    tick(nowMs) {
        if (this.lastTick > 0) {
            this.samples.push(nowMs - this.lastTick);
            const cutoff = nowMs - this.windowMs;
            // Drop frames that fall outside the moving window.
            while (this.samples.length > 0 && nowMs - this.samples[0] > cutoff) {
                this.samples.shift();
            }
        }
        this.lastTick = nowMs;
        this.render();
    }
    render() {
        const n = this.samples.length;
        let fps = 0;
        if (n > 0) {
            let sum = 0;
            for (const dt of this.samples)
                sum += dt;
            const avg = sum / n;
            fps = avg > 0 ? 1000 / avg : 0;
        }
        let line = `fps ${fps.toFixed(1)}`;
        const mem = performance.memory;
        if (mem) {
            line += `  mem ${(mem.usedJSHeapSize / 1024 / 1024).toFixed(1)}MB`;
        }
        this.el.textContent = line;
    }
    /** Remove the HUD element from the DOM. */
    destroy() {
        this.el.remove();
    }
}
