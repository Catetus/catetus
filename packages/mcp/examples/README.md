# Catetus MCP — paste-ready client configs

This directory has one config file per major MCP client. Open the one for your
editor, copy the relevant block into the right config file on your machine,
swap `cat_live_REPLACE_ME` for your real API key (or omit the env block entirely
to use the free tier), and reload.

Every config gives you the same Catetus toolset — only the host changes.

---

## Tier matrix

| Tier | How to enable | Tools exposed |
|---|---|---|
| **Free** | Install with no API key (any stdio config below; just omit the `env` block or leave the placeholder unset) | `analyze`, `list_presets`, `list_scenes`, `get_scene`, `optimize`, `compare`, `list_competitor_codecs`, `validate_pipeline` |
| **Paid** | Set `CATETUS_API_KEY=cat_live_…` on stdio, **or** `Authorization: Bearer cat_live_…` on HTTP | all of the above **plus** `encode`, `score_fidelity`, `repack`, `predict_quality`, `recommend_preset`, `batch_jobs`, `list_jobs` |

Get a key at <https://catetus.com/dashboard>. Pricing: <https://catetus.com/pricing>.

---

## Transport matrix

| Variant | Endpoint | Pros | Cons |
|---|---|---|---|
| **stdio (npx)** | `npx @catetus/mcp@latest` | Works offline for free tools; bundles SplatBench corpus; no Node required at runtime if you use the MCPB bundle | Requires Node 20+ unless using MCPB |
| **HTTP (hosted)** | `https://mcp.catetus.com` | Zero local install; always latest server version; required for clients without stdio support | Paid tier only; needs network |

For paid usage, both transports work and bill against the same key. Pick whichever fits your client; HTTP is more convenient on remote/cloud editors.

---

## Per-client setup

| Client | File in this directory | Config file on your machine |
|---|---|---|
| **Claude Desktop** | [`claude-desktop.json`](./claude-desktop.json) | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), `%APPDATA%\Claude\claude_desktop_config.json` (Windows), `~/.config/Claude/claude_desktop_config.json` (Linux) |
| **Claude Code (CLI)** | [`claude-code.json`](./claude-code.json) | `~/.claude.json` (global) or `<project>/.mcp.json` (per-project). The `claude mcp add` command is the easiest way to edit these. |
| **Cursor** | [`cursor.json`](./cursor.json) | `~/.cursor/mcp.json` (global) or `<project>/.cursor/mcp.json` (per-project) |
| **Cline (VS Code)** | [`cline.json`](./cline.json) | `cline_mcp_settings.json` — open via the Cline sidebar → "MCP Servers" → "Configure MCP Servers" |
| **Zed** | [`zed.json`](./zed.json) | `~/.config/zed/settings.json` (under the `context_servers` key) |
| **Continue.dev** | [`continue.json`](./continue.json) | `~/.continue/config.json` (legacy JSON) or `~/.continue/config.yaml` (current) |

Every file in this directory is strict JSON (parses with `JSON.parse`). The
`_comment` keys and the `*-disabled` server-name entries are explanatory
content the host will ignore (no MCP client treats `_comment` as a server) — but
to keep your real config tidy, **delete the `_comment` keys and the
`*-disabled` variants before pasting**. Only keep the one server entry you
want to enable, and rename it to `catetus`.

---

## The fast path: Claude Desktop one-click

If you only want Claude Desktop, **don't bother with this directory** — grab the
`.mcpb` bundle from the latest GitHub release and drag it onto Claude Desktop.
That installs the server with a native install dialog (and prompts for your API
key if you have one).

See <https://github.com/catetus/catetus-mcp/releases> for the latest `.mcpb`.

---

## Troubleshooting

### "Tool not found: encode"
You're on the free tier. Set `CATETUS_API_KEY` (stdio) or `Authorization: Bearer`
(HTTP) and restart. The server filters `tools/list` by tier — paid tools are
hidden, not just locked.

### "npx: command not found" on Claude Desktop (Windows)
Claude Desktop on Windows doesn't always find `npx` on PATH. Either install the
`.mcpb` bundle (which ships Node), or set `command` to the absolute path:
`C:\\Program Files\\nodejs\\npx.cmd`.

### "EACCES" / permission denied on first launch
Some clients sandbox spawned binaries. Make sure your user can run `npx` from
your shell. On macOS, you may need to allow Claude Desktop to spawn helper
processes in System Settings → Privacy & Security.

### Server starts but `tools/list` is empty
You're hitting an HTTP `mcp.catetus.com` endpoint without an `Authorization`
header. The hosted endpoint requires a paid key; the public/free tier is only
available over stdio (and lives in the npm/MCPB packages). Switch to the stdio
variant in your config.

### How to verify it's working
Use the MCP Inspector — it works against any MCP server, including this one:

```bash
npx @modelcontextprotocol/inspector npx -y @catetus/mcp@latest
```

You should see all 8 free tools listed (15 if you pass `--env CATETUS_API_KEY=…`).

---

## Why two transports?

`stdio` is the MCP standard for local-first clients; it runs the server in your
own process tree, your inputs never leave your machine for the free tools, and
it's the only way to get the local-binary mode (analyze large PLYs in-process).

`HTTP (mcp.catetus.com)` is for hosted/remote clients (Claude.ai connectors,
Cursor's remote MCP, Continue's streamable-http) and for users who don't want
to install Node. Paid only — the public tier doesn't make sense to host
because every public tool either reads bundled data or hits the local disk.

The npm package serves both with the same code — `--transport stdio` (default)
or `--transport http --port 3000`.
