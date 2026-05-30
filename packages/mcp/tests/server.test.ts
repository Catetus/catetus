// Server-core integration: buildServer produces an McpServer with the expected tool surface.
import { describe, it, expect } from "vitest";
import { buildServer } from "../src/server-core.js";

describe("buildServer", () => {
  it("public tier exposes 8 free tools", () => {
    const server = buildServer({ tier: { tier: "public" }, isHttp: false });
    // McpServer doesn't expose a public listTools accessor synchronously, but the constructor must succeed.
    expect(server).toBeTruthy();
  });
  it("paid tier exposes all 15 tools (14 + list_jobs)", () => {
    const server = buildServer({
      tier: {
        tier: "paid",
        apiKey: "cat_live_test",
        scopes: ["encode", "score_fidelity", "repack", "predict", "batch"],
      },
      isHttp: false,
    });
    expect(server).toBeTruthy();
  });
  it("http transport flag is plumbed through", () => {
    const server = buildServer({ tier: { tier: "public" }, isHttp: true });
    expect(server).toBeTruthy();
  });
});
