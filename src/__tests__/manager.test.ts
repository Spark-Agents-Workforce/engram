import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ResolvedConfig } from "../config.js";
import { createEngramManager } from "../manager.js";
import { createIndexManager } from "../store.js";
import { createSyncManager } from "../sync.js";
import type { EmbeddingClient, IndexManager } from "../types.js";

interface Harness {
  tempDir: string;
  workspaceDir: string;
  index: IndexManager;
  embedding: EmbeddingClient;
  syncManager: ReturnType<typeof createSyncManager>;
}

const cleanups: Array<() => void> = [];

const baseConfig: ResolvedConfig = {
  geminiApiKey: undefined,
  dimensions: 3,
  chunkTokens: 64,
  chunkOverlap: 0.2,
  reranking: true,
  timeDecay: {
    enabled: true,
    halfLifeDays: 30,
  },
  maxSessionShare: 0.4,
  multimodal: {
    enabled: false,
    modalities: ["image", "audio"],
    maxFileBytes: 10 * 1024 * 1024,
  },
};

function createMockEmbedding(params?: {
  embedText?: EmbeddingClient["embedText"];
  embedBatch?: EmbeddingClient["embedBatch"];
}): EmbeddingClient {
  return {
    dimensions: 3,
    model: "gemini-embedding-2-preview",
    embedText: params?.embedText ?? (async () => new Float32Array([1, 0, 0])),
    embedBatch:
      params?.embedBatch ??
      (async (texts) => texts.map(() => new Float32Array([1, 0, 0]))),
    supportsMultimodal: false,
  };
}

function createHarness(embedding: EmbeddingClient = createMockEmbedding()): Harness {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-manager-"));
  const workspaceDir = path.join(tempDir, "workspace");
  fs.mkdirSync(workspaceDir, { recursive: true });

  const index = createIndexManager({
    dbPath: path.join(tempDir, "index.sqlite"),
    dimensions: embedding.dimensions,
    workspaceDir,
  });

  const syncManager = createSyncManager({
    workspaceDir,
    index,
    embedding,
    chunkTokens: baseConfig.chunkTokens,
    chunkOverlap: baseConfig.chunkOverlap,
  });

  cleanups.push(() => {
    syncManager.close();
    index.close();
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  return { tempDir, workspaceDir, index, embedding, syncManager };
}

function writeWorkspaceFile(workspaceDir: string, relPath: string, content: string): void {
  const absPath = path.join(workspaceDir, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf8");
}

afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

describe("createEngramManager", () => {
  it("status() returns expected fields", async () => {
    const harness = createHarness();
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# Memory\nalpha");

    const manager = createEngramManager({
      index: harness.index,
      embedding: harness.embedding,
      syncManager: harness.syncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    await manager.sync();
    const status = manager.status();

    expect(status.backend).toBe("engram");
    expect(status.provider).toBe("gemini");
    expect(status.model).toBe("gemini-embedding-2-preview");
    expect(status.files).toBe(1);
    expect(status.chunks).toBeGreaterThan(0);
    expect(status.dirty).toBe(false);
    expect(status.workspaceDir).toBe(harness.workspaceDir);
    expect(status.dbPath).toContain("index.sqlite");
    expect(status.sources.some((row) => row.source === "memory")).toBe(true);
    expect(status.vector.enabled).toBeTypeOf("boolean");
    expect(status.vector.dims).toBeTypeOf("number");
    expect(status.fts.enabled).toBe(true);
    expect(status.fts.available).toBe(true);
    expect(status.custom.rrfK).toBe(60);
    expect(status.custom.vectorWeight).toBe(0.7);
    expect(status.custom.bm25Weight).toBe(0.3);
    expect(status.custom.chunkTokens).toBe(baseConfig.chunkTokens);
    expect(status.custom.chunkOverlap).toBe(baseConfig.chunkOverlap);
  });

  it("probeEmbeddingAvailability() returns ok=true when embedding works", async () => {
    const harness = createHarness(createMockEmbedding());
    const manager = createEngramManager({
      index: harness.index,
      embedding: harness.embedding,
      syncManager: harness.syncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    await expect(manager.probeEmbeddingAvailability()).resolves.toEqual({ ok: true });
  });

  it("probeEmbeddingAvailability() returns ok=false and error when embedding fails", async () => {
    const harness = createHarness(
      createMockEmbedding({
        embedText: async () => {
          throw new Error("embedding down");
        },
      }),
    );
    const manager = createEngramManager({
      index: harness.index,
      embedding: harness.embedding,
      syncManager: harness.syncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    const result = await manager.probeEmbeddingAvailability();
    expect(result.ok).toBe(false);
    expect(result.error).toContain("embedding down");
  });

  it("probeVectorAvailability() returns a boolean", async () => {
    const harness = createHarness();
    const manager = createEngramManager({
      index: harness.index,
      embedding: harness.embedding,
      syncManager: harness.syncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    const available = await manager.probeVectorAvailability();
    expect(typeof available).toBe("boolean");
  });

  it("search() delegates to search pipeline and returns results", async () => {
    const harness = createHarness();
    harness.index.indexFile(
      {
        fileKey: "memory/alpha.md",
        contentHash: "hash-alpha",
        source: "memory",
        indexedAt: Date.now(),
      },
      [
        {
          text: "alpha semantic token",
          startLine: 1,
          endLine: 1,
          hash: "alpha-1",
        },
      ],
      [new Float32Array([1, 0, 0])],
    );

    const manager = createEngramManager({
      index: harness.index,
      embedding: harness.embedding,
      syncManager: harness.syncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    const results = await manager.search("alpha", { maxResults: 5 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0]?.path).toBe("memory/alpha.md");
  });

  it("readFile() delegates to index.readFileContent()", async () => {
    const harness = createHarness();
    writeWorkspaceFile(harness.workspaceDir, "memory/readme.md", "line1\nline2\nline3");

    const manager = createEngramManager({
      index: harness.index,
      embedding: harness.embedding,
      syncManager: harness.syncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    const result = await manager.readFile({
      relPath: "memory/readme.md",
      from: 2,
      lines: 2,
    });

    expect(result.text).toBe("line2\nline3");
    expect(result.path).toBe("memory/readme.md");
  });

  it("close() closes index and sync manager", async () => {
    const harness = createHarness();
    const indexClose = vi.fn(() => harness.index.close());
    const syncClose = vi.fn(() => harness.syncManager.close());

    const wrappedIndex: IndexManager = {
      ...harness.index,
      close: indexClose,
    };

    const wrappedSyncManager = {
      ...harness.syncManager,
      close: syncClose,
    };

    const manager = createEngramManager({
      index: wrappedIndex,
      embedding: harness.embedding,
      syncManager: wrappedSyncManager,
      config: baseConfig,
      workspaceDir: harness.workspaceDir,
    });

    await manager.close();

    expect(indexClose).toHaveBeenCalledTimes(1);
    expect(syncClose).toHaveBeenCalledTimes(1);
  });
});
