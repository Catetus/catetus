import { defineConfig } from "astro/config";

// Static-only output. Deploys identically to Vercel, Cloudflare Pages, GitHub Pages.
export default defineConfig({
  site: "https://catetus.com",
  output: "static",
  build: {
    format: "directory",
  },
  trailingSlash: "ignore",
  devToolbar: { enabled: false },
});
