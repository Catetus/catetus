// Smoke test: spin up an in-process MCP server, register the resources
// subsystem, drive a client over an InMemoryTransport, and verify that
//   - resources/list returns all 7 static resources (by URI + mimeType)
//   - resources/templates/list returns the catetus://scene/{scene_id} template
//   - resources/read works for each static URI (returns non-empty content with
//     the declared mimeType)
//   - the scene template resolves a known canonical-11 scene_id
//   - the scene template returns an error payload for an unknown scene_id

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

import {
  registerAllResources,
  RESOURCE_URIS,
  STATIC_RESOURCE_URIS,
  RESOURCE_TEMPLATE_URIS,
} from "../src/resources/index.js";

let client: Client;
let server: McpServer;

beforeAll(async () => {
  server = new McpServer(
    { name: "catetus-resources-test", version: "0.0.0-test" },
    { capabilities: { resources: { subscribe: true, listChanged: true } } },
  );
  registerAllResources(server);

  client = new Client(
    { name: "catetus-resources-test-client", version: "0.0.0-test" },
    { capabilities: {} },
  );

  const [clientTx, serverTx] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTx), client.connect(clientTx)]);
});

afterAll(async () => {
  await client.close();
  await server.close();
});

describe("resources/list", () => {
  it("returns all 7 static resources (plus template-listed scenes)", async () => {
    const result = await client.listResources();
    expect(STATIC_RESOURCE_URIS).toHaveLength(7);

    // The SDK merges static resources + the template's list() output. Filter to
    // just the 7 fixed URIs and assert the set matches exactly.
    const staticSet = new Set<string>(STATIC_RESOURCE_URIS);
    const staticReturned = result.resources
      .map((r) => r.uri)
      .filter((u) => staticSet.has(u))
      .sort();
    expect(staticReturned).toEqual([...STATIC_RESOURCE_URIS].sort());
  });

  it("each static resource has a title, description, mimeType, and annotations", async () => {
    const result = await client.listResources();
    const staticSet = new Set<string>(STATIC_RESOURCE_URIS);
    const staticResources = result.resources.filter((r) => staticSet.has(r.uri));
    expect(staticResources).toHaveLength(7);
    for (const r of staticResources) {
      expect(r.name, `resource ${r.uri} missing name`).toBeTruthy();
      expect(r.title, `resource ${r.uri} missing title`).toBeTruthy();
      expect(r.description, `resource ${r.uri} missing description`).toBeTruthy();
      expect(r.mimeType, `resource ${r.uri} missing mimeType`).toBeTruthy();
      expect(r.annotations, `resource ${r.uri} missing annotations`).toBeTruthy();
    }
  });

  it("mimeTypes match the spec (json for data, markdown for docs)", async () => {
    const result = await client.listResources();
    const byUri = new Map(result.resources.map((r) => [r.uri, r]));
    const expectedMime: Record<string, string> = {
      [RESOURCE_URIS.canonical11]: "application/json",
      [RESOURCE_URIS.splatbenchV0]: "application/json",
      [RESOURCE_URIS.threeTierComparison]: "text/markdown",
      [RESOURCE_URIS.competitorCodecs]: "application/json",
      [RESOURCE_URIS.catalog]: "application/json",
      [RESOURCE_URIS.sdkTerms]: "text/markdown",
      [RESOURCE_URIS.presetCheatsheet]: "text/markdown",
    };
    for (const [uri, mime] of Object.entries(expectedMime)) {
      expect(byUri.get(uri)?.mimeType, `mime for ${uri}`).toBe(mime);
    }
  });
});

describe("resources/templates/list", () => {
  it("returns the catetus://scene/{scene_id} template", async () => {
    const result = await client.listResourceTemplates();
    expect(result.resourceTemplates).toHaveLength(RESOURCE_TEMPLATE_URIS.length);
    expect(RESOURCE_TEMPLATE_URIS).toHaveLength(1);
    expect(result.resourceTemplates[0].uriTemplate).toBe(RESOURCE_URIS.sceneTemplate);
    expect(result.resourceTemplates[0].mimeType).toBe("application/json");
  });
});

describe("resources/read", () => {
  for (const uri of [
    RESOURCE_URIS.canonical11,
    RESOURCE_URIS.splatbenchV0,
    RESOURCE_URIS.threeTierComparison,
    RESOURCE_URIS.competitorCodecs,
    RESOURCE_URIS.catalog,
    RESOURCE_URIS.sdkTerms,
    RESOURCE_URIS.presetCheatsheet,
  ]) {
    it(`returns non-empty content for ${uri}`, async () => {
      const result = await client.readResource({ uri });
      expect(result.contents).toHaveLength(1);
      const c = result.contents[0];
      expect(c.uri).toBe(uri);
      expect(c.mimeType).toBeTruthy();
      // text content path (we never embed blobs in this package).
      expect(typeof c.text).toBe("string");
      expect((c.text as string).length).toBeGreaterThan(50);
    });
  }

  it("parses JSON resources as valid JSON", async () => {
    const jsonUris = [
      RESOURCE_URIS.canonical11,
      RESOURCE_URIS.splatbenchV0,
      RESOURCE_URIS.competitorCodecs,
      RESOURCE_URIS.catalog,
    ];
    for (const uri of jsonUris) {
      const result = await client.readResource({ uri });
      expect(() => JSON.parse(result.contents[0].text as string)).not.toThrow();
    }
  });

  it("canonical-11 JSON exposes 11 scenes", async () => {
    const result = await client.readResource({ uri: RESOURCE_URIS.canonical11 });
    const data = JSON.parse(result.contents[0].text as string);
    expect(data.scenes).toHaveLength(11);
    expect(data.scenes.map((s: { scene: string }) => s.scene)).toContain("bonsai");
  });

  it("preset catalog exposes 12 presets including v52-quality", async () => {
    const result = await client.readResource({ uri: RESOURCE_URIS.catalog });
    const data = JSON.parse(result.contents[0].text as string);
    expect(data.presets.length).toBeGreaterThanOrEqual(12);
    const names = data.presets.map((p: { name: string }) => p.name);
    expect(names).toContain("v52-quality");
    expect(names).toContain("web-mobile");
    expect(names).toContain("lossless-repack");
  });
});

describe("scene template", () => {
  it("resolves a known canonical-11 scene_id (bonsai)", async () => {
    const result = await client.readResource({ uri: "catetus://scene/bonsai" });
    expect(result.contents).toHaveLength(1);
    const payload = JSON.parse(result.contents[0].text as string);
    expect(payload.scene_id).toBe("bonsai");
    expect(payload.corpus).toBe("canonical-11");
    expect(payload.record.scene).toBe("bonsai");
    expect(payload.leaderboardRef).toBe("catetus://bench/canonical-11#bonsai");
  });

  it("resolves a known splatbench-v0 scene_id", async () => {
    const result = await client.readResource({
      uri: "catetus://scene/splatbench_product_proxy",
    });
    const payload = JSON.parse(result.contents[0].text as string);
    expect(payload.corpus).toBe("splatbench-v0");
    expect(payload.record.id).toBe("splatbench_product_proxy");
  });

  it("returns a scene_not_found error for an unknown scene_id", async () => {
    const result = await client.readResource({ uri: "catetus://scene/does-not-exist" });
    const payload = JSON.parse(result.contents[0].text as string);
    expect(payload.error.code).toBe("scene_not_found");
    expect(payload.error.hint).toContain("bicycle");
  });

  it("template list enumerates all canonical-11 + splatbench-v0 scenes", async () => {
    const result = await client.listResources();
    // The list call returns static resources only — template-listed scenes
    // come through the template's `list` callback during listResources too.
    // The SDK merges them.
    const sceneEntries = result.resources.filter((r) => r.uri.startsWith("catetus://scene/"));
    expect(sceneEntries.length).toBeGreaterThanOrEqual(11 + 16);
    const sceneIds = sceneEntries.map((r) => r.uri.replace("catetus://scene/", ""));
    expect(sceneIds).toContain("bonsai");
    expect(sceneIds).toContain("bicycle");
    expect(sceneIds).toContain("splatbench_product_proxy");
  });
});
