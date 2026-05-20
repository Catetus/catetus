import { shell } from '../chrome';

// Viewer route. The website is shipped as a separate Vite project; the
// viewer-app builds independently to its own `dist/`. The recommended deploy
// pattern is to publish the website at `/` and the viewer-app at `/viewer/`
// on the same origin (CDN rewrite). Until then this route shows usage and a
// link that works whether the viewer is mounted at /viewer/ or hosted
// elsewhere — set `window.__CATETUS_VIEWER_URL__` at build time to
// override (see README).

const DEFAULT_VIEWER_URL = '/viewer/';
// Real photogrammetric scene (cactus, ~186k splats) from steam_studio / 3D Scan
// Studio iris, released CC0 (https://note.com/steam_studio/n/ne9736d94f162).
// Re-encoded through the `web-mobile` preset to 6.7 MB GLB. See
// `experiments/demo-scene-integration/RESULT.md` for the license-provenance
// receipt and capture details.
const DEFAULT_DEMO_SCENE = '/demos/cactus_steam_cc0.glb';

export function renderViewer(): string {
  const viewerUrl =
    (window as unknown as { __CATETUS_VIEWER_URL__?: string })
      .__CATETUS_VIEWER_URL__ ?? DEFAULT_VIEWER_URL;
  // Pre-load a small demo scene so cold visitors see something on /viewer.
  // ?scene=<url> on this page overrides the default.
  const params = new URLSearchParams(window.location.search);
  const demoScene = params.get('scene') ?? DEFAULT_DEMO_SCENE;
  // The viewer-app reads ?src=<url> at boot to auto-fetch.
  const viewerSrc = `${viewerUrl}?src=${encodeURIComponent(demoScene)}`;

  return shell(`
    <section class="hero" style="padding-top:24px">
      <div class="eyebrow">viewer</div>
      <h1 style="font-size:32px;">Drop a file.</h1>
      <p class="lede">
        Reads Inria <code>.ply</code>, antimatter15 <code>.splat</code>,
        PlayCanvas <code>.sog</code>, and Catetus <code>.glb</code>
        (with optional <code>.glb.shpal</code> palette sidecar). WebGL2.
        Local — files never leave your browser.
      </p>
      <p class="caption" style="opacity:0.6;margin-top:4px">
        Preloaded with <code>${demoScene}</code> &mdash;
        CC0 / public domain, captured by
        <a href="https://www.steam-studio.jp" rel="noopener" target="_blank">steam-studio.jp</a>
        (3D Scan Studio iris), re-encoded through Catetus <code>web-mobile</code>.
      </p>
    </section>

    <div class="viewer-frame">
      <iframe
        src="${viewerSrc}"
        title="Catetus viewer"
        allow="fullscreen"
        sandbox="allow-scripts allow-same-origin allow-downloads"
      ></iframe>
    </div>
    <p class="caption" style="margin-top:10px">
      iframe → <code>${viewerUrl}</code>. If you see a blank frame, the
      viewer-app is not yet co-hosted on this origin.
      Build it from <code>packages/viewer-app/</code> and serve <code>dist/</code>
      under <code>/viewer/</code>, or set <code>window.__CATETUS_VIEWER_URL__</code>
      at build time. See <a href="https://github.com/Catetus/catetus/tree/main/packages/website" rel="noopener" target="_blank">README</a>.
    </p>

    <section class="section">
      <h2>What it does, exactly</h2>
      <ul class="prose">
        <li>Decodes <code>.ply</code> (Inria official 3DGS export, SH up to L3).</li>
        <li>Decodes <code>.splat</code> (antimatter15 binary, SH L0 only).</li>
        <li>Decodes <code>.sog</code> (PlayCanvas zip-of-WebP container).</li>
        <li>Decodes <code>.glb</code> + optional <code>.glb.shpal</code> SH-rest palette.</li>
        <li>Reads V5.2 joint-tail sidecars (<code>.glb.v5tail</code> / <code>.sog.v5tail</code>) and fuses them at decode time.</li>
        <li>Orbit, free-look, pan, fly. Camera-state JSON export.</li>
      </ul>
      <p class="caption">
        Source: <code>packages/viewer-app/</code> &middot;
        <a href="https://github.com/Catetus/catetus/tree/main/packages/viewer-app" rel="noopener" target="_blank">browse on GitHub</a>
      </p>
    </section>
  `);
}
