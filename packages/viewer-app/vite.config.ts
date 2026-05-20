import { defineConfig } from 'vite';

// Static SPA. Cloudflare Pages / Vercel / GitHub Pages all consume `dist/` as-is.
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
    outDir: 'dist',
    assetsInlineLimit: 0,
    sourcemap: true,
  },
  worker: { format: 'es' },
  server: { port: 5173, host: true },
});
