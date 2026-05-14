import { defineConfig } from "astro/config";

// Static-only output. Deploys identically to Vercel, Cloudflare Pages, GitHub Pages.
export default defineConfig({
  site: "https://splatforge.dev",
  output: "static",
  build: {
    format: "directory",
  },
  trailingSlash: "ignore",
  devToolbar: { enabled: false },
});
