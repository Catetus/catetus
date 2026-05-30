# dist-bundle/

Generated MCPB artifacts land here. **Everything in this directory except this
README is gitignored.**

## Contents (after a successful build)

```
dist-bundle/
├── README.md                       # this file
├── build/                          # staging dir for `mcpb pack` (regenerated each run)
│   ├── manifest.json
│   ├── icon.png                    # if you added one at packages/mcp/icon.png
│   └── server/
│       ├── server-stdio.js         # entry point (from packages/mcp/dist/)
│       ├── server-core.js          # shared registration code
│       ├── tools/, resources/, ... # transpiled TS output tree
│       └── node_modules/           # bundled deps (so Claude Desktop's node can resolve them)
└── catetus-<version>.mcpb          # the shippable bundle (zip archive)
```

## How to (re)build

From the repo root:

```bash
# Build the TS server first, then the bundle
( cd packages/mcp && npm run build )
bash packages/mcp/scripts/build-mcpb.sh

# Or do both in one shot:
bash packages/mcp/scripts/build-mcpb.sh --build
```

Result: `packages/mcp/dist-bundle/catetus-<version>.mcpb`. Drag onto Claude
Desktop to install, or upload to a GitHub release.

## Pre-flight checks

```bash
# Schema validate the manifest without packing
bash packages/mcp/scripts/build-mcpb.sh --validate

# Sign for distribution (requires Apple cert configured for @anthropic-ai/mcpb)
bash packages/mcp/scripts/build-mcpb.sh --sign
```

## Gitignore

Add (or confirm) this line in `packages/mcp/.gitignore`:

```
dist-bundle/build/
dist-bundle/*.mcpb
```

Keep `dist-bundle/README.md` in git so the directory exists for new clones.
