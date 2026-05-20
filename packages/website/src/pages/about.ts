import { shell } from '../chrome';

export function renderAbout(): string {
  return shell(`
    <section class="hero" style="padding-top:24px">
      <div class="eyebrow">about</div>
      <h1 style="font-size:32px;">A measurement layer for a young file format.</h1>
    </section>

    <div class="prose">
      <p>
        3D Gaussian Splatting moved from a single 2023 SIGGRAPH paper to a
        production-shaped ecosystem in eighteen months. Forty-plus
        compression methods exist in the literature, two container formats
        are in real-world use, and almost no two of them agree on numbers.
        Vendors quote PSNR on subsets, with non-disclosed render settings,
        against undisclosed baselines. <strong>Catetus exists to fix the
        bench.</strong> The codecs we ship are evidence the bench works, not
        the product.
      </p>

      <h3>How we measure</h3>
      <p>
        Every quality number on this site is reproduced by the same loop:
        <code>gsplat 1.5.3</code>, a fixed 72-view orbit (24 azimuth × 3
        elevation), 512² render, SH degree 3, float PSNR convention. Scenes
        are the canonical-11 set (Mip-NeRF 360 + Tanks-and-Temples), all from
        the same iter-30k Inria training checkpoints. Bytes are total file
        bytes on disk, no asterisks. The harness is in the repo; every
        result above links back to the experiment directory that produced it.
      </p>

      <h3>Wire-format philosophy</h3>
      <p>
        A new codec that can't be opened by the tools people already use is
        a dead codec. Both of Catetus's compression techniques ride
        inside containers that already exist:
      </p>
      <ul>
        <li><strong>T2.1.R</strong> (Jacobian-weighted Lloyd) is a one-line edit to a K-means centroid update. It produces standard PlayCanvas SOG files; SuperSplat opens them.</li>
        <li><strong>V5.2</strong> (joint-tail residual sidecar) is a detached companion file. Legacy viewers ignore it. Catetus readers fuse it at decode time.</li>
      </ul>
      <p>
        Both designs are documented in the
        <a href="https://github.com/Catetus/catetus/blob/main/experiments/defensive-publication/V5_2_PUBLIC.md" rel="noopener" target="_blank">V5.2 defensive publication</a>
        with enough precision for a third-party decoder.
      </p>

      <h3>Prior art</h3>
      <ul>
        <li>Kerbl et al., <em>3D Gaussian Splatting for Real-Time Radiance Field Rendering</em>, SIGGRAPH 2023.</li>
        <li>Morgenstern et al., <em>Self-Organizing Gaussians</em>, arXiv 2312.13299 — the SOG container format (PlayCanvas / Snap), MIT-licensed.</li>
        <li>GoDe, arXiv 2501.13558 — group-aware Gaussian encoding, an inspiration for the per-cell predictor structure in V5.2.</li>
        <li>Bagdasarian et al., <em>3DGS Compression Survey</em>, arXiv 2407.09510 — the field map we calibrate against.</li>
        <li>MPEG G-PCC — the broader point-cloud compression line that motivates the residual / predictive split in V5.2.</li>
        <li><a href="https://github.com/nerfstudio-project/gsplat" rel="noopener" target="_blank">gsplat</a> — the differentiable rasterizer used end-to-end for benching.</li>
      </ul>

      <h3>What this site is not</h3>
      <p class="meta">
        Not a product. Not a service. No analytics, no contact form, no
        sign-up. A static page describing measurements that anyone can
        reproduce from the public repository. The project is in stealth;
        public release timing and licensing of the Catetus encoders are
        a separate decision.
      </p>
    </div>
  `);
}
