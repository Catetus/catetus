import { defineConfig } from 'vite';

// Static site. Output `dist/` is deployable to any CDN: Cloudflare Pages,
// Vercel (static), GitHub Pages, S3+CloudFront, Netlify, etc.
//
// History-mode (clean URLs like /viewer, /about) requires the host to
// fall back unknown paths to /index.html. See README for per-host notes.
export default defineConfig({
  base: '/',
  build: {
    target: 'es2022',
    outDir: 'dist',
    assetsInlineLimit: 0,
    sourcemap: true,
    cssMinify: true,
  },
  server: { port: 5174, host: true },
  preview: { port: 5174 },
});
