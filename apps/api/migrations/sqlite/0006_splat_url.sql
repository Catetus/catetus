-- Adds a sidecar URL for the antimatter15-format .splat file the worker
-- now uploads alongside the .glb. Used by the homepage TryIt result viewer
-- via the vendored /am15/ viewer for crisp render quality without
-- depending on our custom WebGL2 SDK.
ALTER TABLE jobs ADD COLUMN splat_url TEXT NULL;
