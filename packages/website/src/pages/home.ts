import { shell } from '../chrome';

export function renderHome(): string {
  return shell(`
    <section class="hero">
      <div class="eyebrow">research preview &middot; stealth</div>
      <h1>Trusted <em>measurement and interop</em> for 3D Gaussian Splatting compression.</h1>
      <p class="lede">
        Catetus is a measurement layer and a set of container-compatible
        sidecars for 3DGS. The codecs are proof points. The point is
        reproducible numbers, an open viewer that reads every file you have,
        and wire formats that survive the trip through somebody else's tools.
      </p>
      <div class="cta-row">
        <a class="btn primary" href="#/viewer">Open the viewer</a>
        <a class="btn" href="https://github.com/Catetus/catetus" rel="noopener" target="_blank">GitHub</a>
      </div>
    </section>

    <hr />

    <section class="section">
      <div class="tag">headline result</div>
      <h2>+6.54 dB on bonsai, inside the PlayCanvas SOG container, with receipts.</h2>
      <p class="lede" style="font-size:16px;margin-top:4px;">
        A 747 KB joint-tail sidecar (V5.2) rides on top of any
        <code>.sog</code> file as a <code>.sog.v5tail</code> companion. Legacy
        SOG viewers ignore it; Catetus readers fuse it in. SuperSplat
        compatibility is preserved end-to-end.
      </p>

      <div class="grid-3" style="margin-top:24px">
        <div class="card accent">
          <p class="num">+6.54<span class="unit">dB</span></p>
          <p class="sub">bonsai PSNR lift, V5.2 K=1% sidecar over vanilla SOG.</p>
        </div>
        <div class="card">
          <p class="num">+3.95<span class="unit">%</span></p>
          <p class="sub">total byte overhead. 747 KB on top of an 18.9 MB SOG.</p>
        </div>
        <div class="card">
          <p class="num">SOG<span class="unit">compatible</span></p>
          <p class="sub">SuperSplat opens the base file unchanged. Sidecar is detached.</p>
        </div>
      </div>

      <table class="results" style="margin-top:24px">
        <thead>
          <tr><th>variant</th><th class="num">sidecar</th><th class="num">total</th><th class="num">psnr</th><th class="num">Δ vs vanilla</th></tr>
        </thead>
        <tbody>
          <tr><td>vanilla SOG</td><td class="num">—</td><td class="num">18.9 MB</td><td class="num">47.309 dB</td><td class="num">—</td></tr>
          <tr><td>default_K1pct (magnitude select)</td><td class="num">687 KB</td><td class="num">19.6 MB</td><td class="num">47.309 dB</td><td class="num">+0.001 dB</td></tr>
          <tr class="win"><td><b>default_renderJ_K1pct</b> &nbsp;ship</td><td class="num">747 KB</td><td class="num">19.6 MB</td><td class="num">53.852 dB</td><td class="num delta-pos">+6.543 dB</td></tr>
          <tr class="win"><td>default_renderJ_K5pct &nbsp;aggressive</td><td class="num">3.4 MB</td><td class="num">22.3 MB</td><td class="num">61.228 dB</td><td class="num delta-pos">+13.919 dB</td></tr>
        </tbody>
      </table>
      <p class="caption">
        gsplat 1.5.3, 72-view orbit (24 az × 3 el), 512² float-PSNR.
        Source: <code>experiments/sog-v5tail-retune/RESULT.md</code> &middot;
        <a href="https://github.com/Catetus/catetus/blob/main/experiments/sog-v5tail-retune/RESULT.md" rel="noopener" target="_blank">view on GitHub</a>
      </p>
    </section>

    <section class="section">
      <div class="tag">canonical-11 leaderboard</div>
      <h2>11 / 11 strict wins versus SOG across the canonical scene set.</h2>
      <p class="lede" style="font-size:16px;margin-top:4px;">
        On the canonical-11 pretrained scene set (Mip-NeRF 360 + Tanks-and-Temples,
        Catetus GLB baseline preset <code>wmv-vq45-no-prune-tight</code>),
        every scene wins on both PSNR and bytes against the same container's
        vanilla SOG. The T2.1.R + V5.2 fidelity-upgrade tier adds another
        +17 dB on top at SOG-comparable bytes (9 / 11 scenes measured;
        bicycle + garden encoding).
      </p>

      <div class="grid-3" style="margin-top:18px">
        <div class="card">
          <p class="num">11 / 11</p>
          <p class="sub">strict wins. PSNR up <em>and</em> bytes down on every scene.</p>
        </div>
        <div class="card">
          <p class="num">+1.99<span class="unit">dB</span></p>
          <p class="sub">average PSNR lift across the 11 scenes.</p>
        </div>
        <div class="card">
          <p class="num">−30.6<span class="unit">%</span></p>
          <p class="sub">average byte reduction at higher quality.</p>
        </div>
      </div>
      <p class="caption" style="margin-top:14px">
        Full per-scene table:
        <a href="https://github.com/Catetus/catetus/blob/main/experiments/gaussian-rasterizer-bench/CANONICAL_11_LEADERBOARD.md" rel="noopener" target="_blank">gaussian-rasterizer-bench/CANONICAL_11_LEADERBOARD.md</a>
      </p>
    </section>

    <section class="section">
      <div class="tag">prior art &middot; archived</div>
      <h2>Defensive publication: V5.2 + T2.1.R, specified for third-party decode.</h2>
      <p class="lede" style="font-size:16px;margin-top:4px;">
        The two techniques — Jacobian-weighted Lloyd for VQ palettes (T2.1.R)
        and the render-Jacobian-selected residual sidecar (V5.2) — are
        documented precisely enough to be reimplemented without the
        Catetus codebase. Submitted to arXiv-style archival to establish
        prior art.
      </p>
      <div class="cta-row" style="margin-top:18px">
        <a class="btn" href="https://github.com/Catetus/catetus/blob/main/experiments/defensive-publication/V5_2_PUBLIC.md" rel="noopener" target="_blank">Read V5_2_PUBLIC.md</a>
        <a class="btn" href="https://github.com/Catetus/catetus/blob/main/experiments/defensive-publication/RELEASE_NOTES.md" rel="noopener" target="_blank">Release notes</a>
      </div>
    </section>

    <hr />

    <section class="section">
      <div class="tag">try it</div>
      <h2>The viewer reads everything.</h2>
      <p class="lede" style="font-size:16px;margin-top:4px;">
        Drag and drop an Inria <code>.ply</code>, antimatter15 <code>.splat</code>,
        PlayCanvas <code>.sog</code>, or Catetus <code>.glb + .glb.shpal</code>.
        WebGL2. Runs offline once loaded. No telemetry.
      </p>
      <div class="cta-row" style="margin-top:18px">
        <a class="btn primary" href="#/viewer">Open viewer</a>
      </div>
    </section>
  `);
}
