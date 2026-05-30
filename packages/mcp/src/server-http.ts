#!/usr/bin/env node
// Streamable HTTP transport entrypoint per ARCHITECTURE.md §3.2.
// Stateless mode: a fresh transport per request, no session state.

import express, { type Request, type Response } from "express";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { buildServer } from "./server-core.js";
import { resolveTier } from "./auth.js";
import { CATETUS_MCP_VERSION } from "./version.js";

const SUPPORTED_PROTOCOL_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26"];
const ALLOWED_ORIGINS = new Set([
  "https://claude.ai",
  "https://cursor.com",
  "https://www.cursor.com",
]);

const app = express();
app.use(express.json({ limit: "16mb" }));

// CORS preflight
app.use((req: Request, res: Response, next) => {
  const origin = req.headers.origin;
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader(
      "Access-Control-Allow-Headers",
      "content-type, mcp-protocol-version, mcp-session-id, authorization",
    );
    res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  }
  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }
  next();
});

// Health check (NOT MCP)
app.get("/healthz", (_req: Request, res: Response) => {
  res.json({
    status: "ok",
    version: CATETUS_MCP_VERSION,
    upstream: "unknown",
  });
});

// Discovery
app.get("/.well-known/mcp.json", (_req: Request, res: Response) => {
  res.json({
    name: "catetus",
    version: CATETUS_MCP_VERSION,
    transport: "streamable-http",
    capabilities: {
      tools: true,
      prompts: true,
      resources: true,
    },
  });
});

// MCP endpoint
app.post("/mcp", async (req: Request, res: Response) => {
  // Origin validation (DNS rebinding)
  const origin = req.headers.origin;
  if (origin && !ALLOWED_ORIGINS.has(origin)) {
    res.status(403).json({ error: "forbidden_origin" });
    return;
  }
  // Protocol version validation
  const protoRaw = req.headers["mcp-protocol-version"];
  const proto = Array.isArray(protoRaw) ? protoRaw[0] : protoRaw;
  if (proto && !SUPPORTED_PROTOCOL_VERSIONS.includes(proto)) {
    res.status(400).json({
      error: "unsupported_protocol",
      supported: SUPPORTED_PROTOCOL_VERSIONS,
    });
    return;
  }

  const tier = resolveTier(req.headers as Record<string, string | string[] | undefined>);
  const server = buildServer({ tier, isHttp: true });
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined, // stateless
  });
  res.on("close", () => {
    transport.close();
    server.close();
  });
  await server.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? "0.0.0.0";
app.listen(port, host, () => {
  process.stderr.write(
    `[catetus-mcp] http ready on http://${host}:${port}/mcp (version=${CATETUS_MCP_VERSION})\n`,
  );
});
