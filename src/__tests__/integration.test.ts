import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { search } from "../search.js";
import { createIndexManager } from "../store.js";
import { createSyncManager } from "../sync.js";
import type { EmbeddingClient, IndexManager } from "../types.js";

interface TestHarness {
  tempDir: string;
  workspaceDir: string;
  index: IndexManager;
  sync: ReturnType<typeof createSyncManager>;
  embedding: EmbeddingClient;
}

function createDeterministicEmbedding(dims: number = 8): EmbeddingClient {
  function textToVector(text: string): Float32Array {
    const vec = new Float32Array(dims);
    const hash = createHash("sha256").update(text).digest();
    for (let i = 0; i < dims; i += 1) {
      vec[i] = (hash[i % hash.length] - 128) / 128;
    }

    let norm = 0;
    for (let i = 0; i < dims; i += 1) {
      norm += vec[i] * vec[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let i = 0; i < dims; i += 1) {
        vec[i] /= norm;
      }
    }
    return vec;
  }

  return {
    dimensions: dims,
    model: "test-deterministic",
    embedText: async (text) => textToVector(text),
    embedBatch: async (texts) => texts.map((text) => textToVector(text)),
    supportsMultimodal: false,
  };
}

function writeWorkspaceFile(workspaceDir: string, relPath: string, content: string): void {
  const absPath = path.join(workspaceDir, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf8");
}

function seedWorkspace(workspaceDir: string): void {
  writeWorkspaceFile(workspaceDir, "MEMORY.md", "# Project\n\nEngram is a memory plugin for OpenClaw.");
  writeWorkspaceFile(
    workspaceDir,
    "memory/decisions.md",
    "# Decisions\n\n## Architecture\n\nWe chose SQLite for storage.\n\n## Embedding\n\nWe use Gemini Embedding-2.",
  );
  writeWorkspaceFile(
    workspaceDir,
    "memory/notes.md",
    "# Notes\n\nToday we fixed bug #123 in the chunker.\nThe error was ENOENT on missing files.",
  );
}

function createHarness(): TestHarness {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-integration-"));
  const workspaceDir = path.join(tempDir, "workspace");
  fs.mkdirSync(workspaceDir, { recursive: true });

  const embedding = createDeterministicEmbedding();
  const index = createIndexManager({
    dbPath: path.join(tempDir, "index.sqlite"),
    dimensions: embedding.dimensions,
    workspaceDir,
  });
  const sync = createSyncManager({
    workspaceDir,
    index,
    embedding,
    chunkTokens: 128,
    chunkOverlap: 0.15,
  });

  return { tempDir, workspaceDir, index, sync, embedding };
}

describe("integration: end-to-end pipeline", () => {
  let harness: TestHarness;

  beforeEach(() => {
    harness = createHarness();
  });

  afterEach(() => {
    harness.sync.close();
    harness.index.close();
    fs.rmSync(harness.tempDir, { recursive: true, force: true });
  });

  it("runs full pipeline from sync to search", async () => {
    seedWorkspace(harness.workspaceDir);

    await harness.sync.sync();

    expect(harness.index.stats().files).toBe(3);

    const sqliteResults = await search("SQLite storage", harness.embedding, harness.index, {
      maxResults: 10,
    });
    expect(sqliteResults.some((row) => row.path === "memory/decisions.md")).toBe(true);

    const enoentResults = await search("bug ENOENT", harness.embedding, harness.index, {
      maxResults: 10,
    });
    expect(enoentResults.some((row) => row.path === "memory/notes.md")).toBe(true);

    const pluginResults = await search("memory plugin", harness.embedding, harness.index, {
      maxResults: 10,
    });
    expect(pluginResults.some((row) => row.path === "MEMORY.md")).toBe(true);
  });

  it("supports incremental sync after adding a file", async () => {
    seedWorkspace(harness.workspaceDir);

    await harness.sync.sync();

    writeWorkspaceFile(harness.workspaceDir, "memory/new.md", "# New\n\nThis is a new file about testing.");
    await harness.sync.sync();

    expect(harness.index.stats().files).toBe(4);

    const results = await search("testing", harness.embedding, harness.index, {
      maxResults: 10,
    });
    expect(results.some((row) => row.path === "memory/new.md")).toBe(true);
  });

  it("removes deleted files on sync", async () => {
    seedWorkspace(harness.workspaceDir);

    await harness.sync.sync();
    fs.rmSync(path.join(harness.workspaceDir, "memory/notes.md"));
    await harness.sync.sync();

    expect(harness.index.stats().files).toBe(2);

    const results = await search("ENOENT", harness.embedding, harness.index, {
      maxResults: 10,
    });
    expect(results.find((row) => row.path === "memory/notes.md")).toBeUndefined();
  });

  it("matches memory_get behavior through readFileContent", async () => {
    seedWorkspace(harness.workspaceDir);
    await harness.sync.sync();

    const full = harness.index.readFileContent("MEMORY.md");
    expect(full?.text).toBe("# Project\n\nEngram is a memory plugin for OpenClaw.");

    const line2 = harness.index.readFileContent("MEMORY.md", 2, 1);
    expect(line2?.text).toBe("");

    expect(harness.index.readFileContent("nonexistent.md")).toBeNull();
  });

  it("returns properly formatted search results", async () => {
    seedWorkspace(harness.workspaceDir);
    await harness.sync.sync();

    const results = await search("memory plugin", harness.embedding, harness.index, {
      maxResults: 10,
    });

    expect(results.length).toBeGreaterThan(0);
    expect(results.every((row) => row.source === "memory")).toBe(true);

    const row = results[0];
    expect(row).toBeDefined();
    expect(row).toHaveProperty("path");
    expect(row).toHaveProperty("startLine");
    expect(row).toHaveProperty("endLine");
    expect(row).toHaveProperty("score");
    expect(row).toHaveProperty("snippet");
    expect(row).toHaveProperty("source");
    expect(row).toHaveProperty("citation");
    expect(row.score).toBeGreaterThanOrEqual(0);
    expect(row.score).toBeLessThanOrEqual(1);
    expect(row.citation).toBe(`${row.path}#L${row.startLine}-L${row.endLine}`);
    expect(row.citation).toMatch(/.+#L\d+-L\d+/);
  });

  it("handles empty workspace sync and search", async () => {
    await harness.sync.sync();

    const stats = harness.index.stats();
    expect(stats.files).toBe(0);
    expect(stats.chunks).toBe(0);

    const results = await search("anything", harness.embedding, harness.index, {
      maxResults: 10,
    });
    expect(results).toEqual([]);
  });
});
