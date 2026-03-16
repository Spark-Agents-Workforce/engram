import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createIndexManager } from "../store.js";
import { createSyncManager, flattenSessionJsonl, remapChunkLines } from "../sync.js";
import type { Chunk, EmbeddingClient, IndexManager, MediaModality, SyncProgress } from "../types.js";

interface TestHarness {
  tempDir: string;
  workspaceDir: string;
  sessionsDir: string;
  index: IndexManager;
  sync: ReturnType<typeof createSyncManager>;
}

const cleanups: Array<() => void> = [];

function mockEmbedding(dims = 3): EmbeddingClient {
  return {
    dimensions: dims,
    model: "test",
    embedText: async () => new Float32Array(dims).fill(0.5),
    embedBatch: async (texts) => texts.map(() => new Float32Array(dims).fill(0.5)),
    supportsMultimodal: false,
  };
}

function writeWorkspaceFile(workspaceDir: string, relPath: string, content: string): void {
  const absPath = path.join(workspaceDir, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf8");
}

function writeSessionFile(sessionsDir: string, relPath: string, content: string): void {
  const absPath = path.join(sessionsDir, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf8");
}

function writeWorkspaceBinaryFile(workspaceDir: string, relPath: string, content: Buffer): void {
  const absPath = path.join(workspaceDir, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content);
}

function createHarness(
  embedding: EmbeddingClient = mockEmbedding(),
  multimodal?: {
    enabled: boolean;
    modalities: MediaModality[];
    maxFileBytes?: number;
  },
): TestHarness {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-sync-"));
  const workspaceDir = path.join(tempDir, "workspace");
  const sessionsDir = path.join(tempDir, "sessions");
  fs.mkdirSync(workspaceDir, { recursive: true });
  fs.mkdirSync(sessionsDir, { recursive: true });

  const index = createIndexManager({
    dbPath: path.join(tempDir, "index.sqlite"),
    dimensions: embedding.dimensions,
    workspaceDir,
  });

  const sync = createSyncManager({
    workspaceDir,
    index,
    embedding,
    chunkTokens: 64,
    chunkOverlap: 0.2,
    sessionsDir,
    multimodal,
  });

  cleanups.push(() => {
    sync.close();
    index.close();
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  return { tempDir, workspaceDir, sessionsDir, index, sync };
}

afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

describe("flattenSessionJsonl", () => {
  it("flattens basic jsonl content", () => {
    const content = [
      JSON.stringify({ role: "user", content: "How do I fix the build?" }),
      JSON.stringify({ role: "assistant", content: "Try running npm install first." }),
    ].join("\n");

    const flattened = flattenSessionJsonl(content);

    expect(flattened).toEqual({
      text: "User: How do I fix the build?\nAssistant: Try running npm install first.",
      lineMap: [1, 2],
    });
  });

  it("handles array content blocks", () => {
    const content = JSON.stringify({
      role: "assistant",
      content: [
        { type: "text", text: "First block" },
        { type: "image", url: "ignored" },
        { type: "text", text: "Second block" },
      ],
    });

    const flattened = flattenSessionJsonl(content);

    expect(flattened).toEqual({
      text: "Assistant: First block Second block",
      lineMap: [1],
    });
  });

  it("skips system messages", () => {
    const content = [
      JSON.stringify({ role: "system", content: "You are helpful." }),
      JSON.stringify({ role: "user", content: "Hello" }),
    ].join("\n");

    const flattened = flattenSessionJsonl(content);

    expect(flattened).toEqual({
      text: "User: Hello",
      lineMap: [2],
    });
  });

  it("returns null for empty files", () => {
    expect(flattenSessionJsonl("")).toBeNull();
    expect(flattenSessionJsonl("\n \n")).toBeNull();
  });

  it("handles OpenClaw session format (message wrapper)", () => {
    const content = [
      JSON.stringify({ type: "session", id: "test-session", timestamp: "2026-03-16" }),
      JSON.stringify({ type: "model_change", id: "x", timestamp: "2026-03-16" }),
      JSON.stringify({ type: "message", message: { role: "user", content: [{ type: "text", text: "What is Engram?" }] } }),
      JSON.stringify({ type: "message", message: { role: "assistant", content: [{ type: "text", text: "A memory plugin." }] } }),
      JSON.stringify({ type: "message", message: { role: "system", content: "system prompt" } }),
      JSON.stringify({ type: "custom", id: "y" }),
    ].join("\n");

    const flattened = flattenSessionJsonl(content);

    expect(flattened).toEqual({
      text: "User: What is Engram?\nAssistant: A memory plugin.",
      lineMap: [3, 4],
    });
  });

  it("handles mixed flat and wrapped formats", () => {
    const content = [
      JSON.stringify({ role: "user", content: "flat format" }),
      JSON.stringify({ type: "message", message: { role: "assistant", content: [{ type: "text", text: "wrapped format" }] } }),
    ].join("\n");

    const flattened = flattenSessionJsonl(content);

    expect(flattened).toEqual({
      text: "User: flat format\nAssistant: wrapped format",
      lineMap: [1, 2],
    });
  });
});

describe("remapChunkLines", () => {
  it("remaps line numbers to original jsonl line positions", () => {
    const chunks: Chunk[] = [
      { text: "alpha", startLine: 1, endLine: 1, hash: "a" },
      { text: "beta", startLine: 2, endLine: 3, hash: "b" },
    ];

    remapChunkLines(chunks, [2, 5, 9]);

    expect(chunks).toEqual([
      { text: "alpha", startLine: 2, endLine: 2, hash: "a" },
      { text: "beta", startLine: 5, endLine: 9, hash: "b" },
    ]);
  });
});

describe("createSyncManager", () => {
  it("discovers memory files", async () => {
    const harness = createHarness();
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# Root\nalpha");
    writeWorkspaceFile(harness.workspaceDir, "memory/notes.md", "# Notes\nbeta");

    await harness.sync.sync();

    expect(harness.index.getFileHash("MEMORY.md")).not.toBeNull();
    expect(harness.index.getFileHash("memory/notes.md")).not.toBeNull();
    expect(harness.index.stats().files).toBe(2);
  });

  it("indexes image files when multimodal is enabled", async () => {
    const embedMedia = vi.fn(async () => new Float32Array([0.1, 0.2, 0.3]));
    const harness = createHarness(
      {
        ...mockEmbedding(),
        supportsMultimodal: true,
        embedMedia,
      },
      {
        enabled: true,
        modalities: ["image"],
      },
    );
    writeWorkspaceBinaryFile(harness.workspaceDir, "memory/screenshot.png", Buffer.from([1, 2, 3, 4]));

    await harness.sync.sync();

    expect(embedMedia).toHaveBeenCalledTimes(1);
    expect(embedMedia.mock.calls[0]?.[1]).toBe("image/png");
    expect(harness.index.getFileHash("memory/screenshot.png")).not.toBeNull();
    expect(harness.index.stats().files).toBe(1);

    const row = harness.index.searchBM25("Image file", 5).find((hit) => hit.fileKey === "memory/screenshot.png");
    expect(row).toBeDefined();
    expect(row?.text).toBe("Image file: memory/screenshot.png");
    expect(row?.source).toBe("memory");
  });

  it("skips media files when multimodal is disabled", async () => {
    const embedMedia = vi.fn(async () => new Float32Array([0.1, 0.2, 0.3]));
    const harness = createHarness(
      {
        ...mockEmbedding(),
        supportsMultimodal: true,
        embedMedia,
      },
      {
        enabled: false,
        modalities: ["image"],
      },
    );
    writeWorkspaceBinaryFile(harness.workspaceDir, "memory/screenshot.png", Buffer.from([9, 8, 7]));

    await harness.sync.sync();

    expect(embedMedia).not.toHaveBeenCalled();
    expect(harness.index.getFileHash("memory/screenshot.png")).toBeNull();
    expect(harness.index.stats().files).toBe(0);
  });

  it("skips media files when embedding client does not support multimodal", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const harness = createHarness(mockEmbedding(), {
      enabled: true,
      modalities: ["image"],
    });
    writeWorkspaceBinaryFile(harness.workspaceDir, "memory/screenshot.png", Buffer.from([5, 4, 3]));

    await harness.sync.sync();

    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0]?.[0])).toContain("does not support media");
    expect(harness.index.getFileHash("memory/screenshot.png")).toBeNull();
    expect(harness.index.stats().files).toBe(0);
    warn.mockRestore();
  });

  it("skips unchanged files", async () => {
    const base = mockEmbedding();
    const embedBatch = vi.fn(base.embedBatch);
    const harness = createHarness({
      ...base,
      embedBatch,
    });
    writeWorkspaceFile(harness.workspaceDir, "memory/unchanged.md", "# Stable\nsame");

    await harness.sync.sync();
    await harness.sync.sync();

    expect(embedBatch).toHaveBeenCalledTimes(1);
  });

  it("re-indexes changed files", async () => {
    const base = mockEmbedding();
    const embedBatch = vi.fn(base.embedBatch);
    const harness = createHarness({
      ...base,
      embedBatch,
    });
    const relPath = "memory/changed.md";
    writeWorkspaceFile(harness.workspaceDir, relPath, "# V1\ncontent");

    await harness.sync.sync();
    writeWorkspaceFile(harness.workspaceDir, relPath, "# V2\nupdated");
    await harness.sync.sync();

    expect(embedBatch).toHaveBeenCalledTimes(2);
  });

  it("removes deleted files", async () => {
    const harness = createHarness();
    const relPath = "memory/delete.md";
    writeWorkspaceFile(harness.workspaceDir, relPath, "# Delete\nsoon");

    await harness.sync.sync();
    expect(harness.index.getFileHash(relPath)).not.toBeNull();
    expect(harness.index.stats().files).toBe(1);

    fs.rmSync(path.join(harness.workspaceDir, relPath));
    await harness.sync.sync();

    expect(harness.index.getFileHash(relPath)).toBeNull();
    expect(harness.index.stats().files).toBe(0);
  });

  it("supports force mode", async () => {
    const base = mockEmbedding();
    const embedBatch = vi.fn(base.embedBatch);
    const harness = createHarness({
      ...base,
      embedBatch,
    });
    writeWorkspaceFile(harness.workspaceDir, "memory/force.md", "# Force\nsame");

    await harness.sync.sync();
    await harness.sync.sync({ force: true });

    expect(embedBatch).toHaveBeenCalledTimes(2);
  });

  it("handles empty workspace", async () => {
    const harness = createHarness();

    await expect(harness.sync.sync()).resolves.toBeUndefined();
    expect(harness.index.stats().files).toBe(0);
    expect(harness.index.stats().chunks).toBe(0);
  });

  it("reports progress callback updates", async () => {
    const harness = createHarness();
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# A\none");
    writeWorkspaceFile(harness.workspaceDir, "memory/b.md", "# B\ntwo");

    const updates: SyncProgress[] = [];
    const progress = vi.fn((update: SyncProgress) => {
      updates.push(update);
    });

    await harness.sync.sync({ progress });

    expect(progress).toHaveBeenCalled();
    expect(updates.length).toBe(2);
    expect(updates[0]?.completed).toBe(1);
    expect(updates[1]?.completed).toBe(2);
    expect(updates[1]?.total).toBe(2);
  });

  it("indexes session jsonl files with sessions source", async () => {
    const harness = createHarness();
    writeSessionFile(
      harness.sessionsDir,
      "session-a.jsonl",
      [
        JSON.stringify({ role: "user", content: "How do I fix the build?" }),
        JSON.stringify({ role: "system", content: "Ignore this system message." }),
        JSON.stringify({ role: "assistant", content: "Try running npm install first." }),
      ].join("\n"),
    );

    await harness.sync.sync();

    expect(harness.index.getFileHash("sessions/session-a.jsonl")).not.toBeNull();

    const sessionStats = new Map(harness.index.stats().sources.map((row) => [row.source, row]));
    expect(sessionStats.get("sessions")?.files).toBe(1);
    expect(sessionStats.get("sessions")?.chunks).toBeGreaterThan(0);

    const results = harness.index.searchBM25("npm install", 10);
    const sessionHit = results.find((row) => row.fileKey === "sessions/session-a.jsonl");
    expect(sessionHit).toBeDefined();
    expect(sessionHit?.source).toBe("sessions");
    expect(sessionHit?.startLine).toBe(1);
    expect(sessionHit?.endLine).toBe(3);
  });

  it("incrementally re-indexes changed session jsonl files", async () => {
    const base = mockEmbedding();
    const embedBatch = vi.fn(base.embedBatch);
    const harness = createHarness({
      ...base,
      embedBatch,
    });
    const relPath = "session-b.jsonl";

    writeSessionFile(
      harness.sessionsDir,
      relPath,
      [
        JSON.stringify({ role: "user", content: "Old content" }),
        JSON.stringify({ role: "assistant", content: "Legacy answer" }),
      ].join("\n"),
    );

    await harness.sync.sync();
    await harness.sync.sync();

    writeSessionFile(
      harness.sessionsDir,
      relPath,
      [
        JSON.stringify({ role: "user", content: "New content" }),
        JSON.stringify({ role: "assistant", content: "Updated answer" }),
      ].join("\n"),
    );
    await harness.sync.sync();

    expect(embedBatch).toHaveBeenCalledTimes(2);
    expect(
      harness.index.searchBM25("Updated answer", 10).some((row) => row.fileKey === "sessions/session-b.jsonl"),
    ).toBe(true);
    expect(
      harness.index.searchBM25("Legacy answer", 10).some((row) => row.fileKey === "sessions/session-b.jsonl"),
    ).toBe(false);
  });

  it("tracks isDirty and markDirty", async () => {
    const harness = createHarness();
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# Initial\ncontent");

    expect(harness.sync.isDirty()).toBe(true);

    await harness.sync.sync();
    expect(harness.sync.isDirty()).toBe(false);

    harness.sync.markDirty();
    expect(harness.sync.isDirty()).toBe(true);

    await harness.sync.sync();
    expect(harness.sync.isDirty()).toBe(false);
  });

  it("warmSession syncs once per session key", async () => {
    const base = mockEmbedding();
    const embedBatch = vi.fn(base.embedBatch);
    const harness = createHarness({
      ...base,
      embedBatch,
    });
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# Warm\ncontent");

    await harness.sync.warmSession("session-a");
    await harness.sync.warmSession("session-a");

    expect(embedBatch).toHaveBeenCalledTimes(1);
    expect(harness.sync.isDirty()).toBe(false);
  });

  it("syncIfDirty only syncs when dirty", async () => {
    const base = mockEmbedding();
    const embedBatch = vi.fn(base.embedBatch);
    const harness = createHarness({
      ...base,
      embedBatch,
    });
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# Search\ncontent");

    await harness.sync.sync();
    harness.sync.syncIfDirty();
    await new Promise((resolve) => {
      setTimeout(resolve, 25);
    });
    expect(embedBatch).toHaveBeenCalledTimes(1);

    // Modify the file so the hash changes, then mark dirty
    writeWorkspaceFile(harness.workspaceDir, "MEMORY.md", "# Search\nupdated content");
    harness.sync.markDirty();
    harness.sync.syncIfDirty();
    for (let i = 0; i < 20; i += 1) {
      if (embedBatch.mock.calls.length === 2) {
        break;
      }
      await new Promise((resolve) => {
        setTimeout(resolve, 10);
      });
    }
    expect(embedBatch).toHaveBeenCalledTimes(2);
  });

  it("close cleans up watcher/timers without errors", () => {
    const harness = createHarness();
    harness.sync.startWatching({
      debounceMs: 10,
      intervalMinutes: 1,
    });

    expect(() => harness.sync.close()).not.toThrow();
    expect(() => harness.sync.close()).not.toThrow();
  });

  it("startWatching skips session watcher when sessionsDir is undefined or missing", () => {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-sync-watch-"));
    const workspaceDir = path.join(tempDir, "workspace");
    fs.mkdirSync(workspaceDir, { recursive: true });

    const withoutSessionsIndex = createIndexManager({
      dbPath: path.join(tempDir, "without-sessions.sqlite"),
      dimensions: 3,
      workspaceDir,
    });
    const withoutSessionsSync = createSyncManager({
      workspaceDir,
      index: withoutSessionsIndex,
      embedding: mockEmbedding(),
      chunkTokens: 64,
      chunkOverlap: 0.2,
    });

    const missingSessionsIndex = createIndexManager({
      dbPath: path.join(tempDir, "missing-sessions.sqlite"),
      dimensions: 3,
      workspaceDir,
    });
    const missingSessionsDir = path.join(tempDir, "missing-sessions");
    const missingSessionsSync = createSyncManager({
      workspaceDir,
      index: missingSessionsIndex,
      embedding: mockEmbedding(),
      chunkTokens: 64,
      chunkOverlap: 0.2,
      sessionsDir: missingSessionsDir,
    });

    cleanups.push(() => {
      withoutSessionsSync.close();
      withoutSessionsIndex.close();
      missingSessionsSync.close();
      missingSessionsIndex.close();
      fs.rmSync(tempDir, { recursive: true, force: true });
    });

    expect(() =>
      withoutSessionsSync.startWatching({
        debounceMs: 10,
        intervalMinutes: 0,
      }),
    ).not.toThrow();
    expect(() =>
      missingSessionsSync.startWatching({
        debounceMs: 10,
        intervalMinutes: 0,
      }),
    ).not.toThrow();
  });
});
