#!/usr/bin/env node
// Stdio transport entrypoint per ARCHITECTURE.md §3.1.
// Reads CATETUS_API_KEY from env to gate paid-tier visibility.

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { buildServer } from "./server-core.js";
import { resolveTier } from "./auth.js";

async function main() {
  const tier = resolveTier({}, process.env);
  const server = buildServer({ tier, isHttp: false });
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // Log to stderr — never stdout (corrupts JSON-RPC).
  process.stderr.write(
    `[catetus-mcp] stdio ready; tier=${tier.tier}; pid=${process.pid}\n`,
  );
}

main().catch((err) => {
  process.stderr.write(`[catetus-mcp] fatal: ${(err as Error).stack ?? err}\n`);
  process.exit(1);
});
