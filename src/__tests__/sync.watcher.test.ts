import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createIndexManager } from "../store.js";
import type { EmbeddingClient } from "../types.js";

type WatchEvent = "add" | "change" | "unlink";

class MockFsWatcher {
  readonly close = vi.fn(async () => {});
  private readonly listeners = new Map<WatchEvent, Array<() => void>>([
    ["add", []],
    ["change", []],
    ["unlink", []],
  ]);

  on(event: WatchEvent, listener: () => void): this {
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.push(listener);
    }
    return this;
  }

  emit(event: WatchEvent): void {
    const handlers = this.listeners.get(event) ?? [];
    for (const handler of handlers) {
      handler();
    }
  }
}

function mockEmbedding(dims = 3): EmbeddingClient {
  return {
    dimensions: dims,
    model: "test",
    embedText: async () => new Float32Array(dims).fill(0.5),
    embedBatch: async (texts) => texts.map(() => new Float32Array(dims).fill(0.5)),
    supportsMultimodal: false,
  };
}

function writeSessionFile(sessionsDir: string, relPath: string, content: string): void {
  const absPath = path.join(sessionsDir, relPath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf8");
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("createSyncManager watcher wiring", () => {
  it("registers a session watcher and schedules sync for session file events", async () => {
    const watchCalls: Array<{ pattern: unknown; options: unknown; watcher: MockFsWatcher }> = [];
    const watchMock = vi.fn((pattern: unknown, options: unknown) => {
      const watcher = new MockFsWatcher();
      watchCalls.push({ pattern, options, watcher });
      return watcher;
    });

    vi.doMock("chokidar", () => ({
      watch: watchMock,
    }));

    const { createSyncManager } = await import("../sync.js");

    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-sync-watch-"));
    const workspaceDir = path.join(tempDir, "workspace");
    const sessionsDir = path.join(tempDir, "sessions");
    fs.mkdirSync(workspaceDir, { recursive: true });
    fs.mkdirSync(sessionsDir, { recursive: true });

    const index = createIndexManager({
      dbPath: path.join(tempDir, "index.sqlite"),
      dimensions: 3,
      workspaceDir,
    });

    const sync = createSyncManager({
      workspaceDir,
      index,
      embedding: mockEmbedding(),
      chunkTokens: 64,
      chunkOverlap: 0.2,
      sessionsDir,
    });

    await sync.sync();
    expect(sync.isDirty()).toBe(false);

    sync.startWatching({
      debounceMs: 20,
      intervalMinutes: 0,
    });

    expect(watchMock).toHaveBeenCalledTimes(2);
    expect(watchCalls[1]?.pattern).toBe("*.jsonl");
    expect((watchCalls[1]?.options as { cwd?: string } | undefined)?.cwd).toBe(sessionsDir);

    writeSessionFile(
      sessionsDir,
      "watched-session.jsonl",
      [
        JSON.stringify({ role: "user", content: "new user prompt" }),
        JSON.stringify({ role: "assistant", content: "new assistant reply" }),
      ].join("\n"),
    );
    watchCalls[1]?.watcher.emit("add");

    const deadline = Date.now() + 1_000;
    while (Date.now() < deadline) {
      if (index.getFileHash("sessions/watched-session.jsonl") !== null) {
        break;
      }
      await new Promise((resolve) => {
        setTimeout(resolve, 25);
      });
    }

    expect(index.getFileHash("sessions/watched-session.jsonl")).not.toBeNull();

    sync.close();
    expect(watchCalls[0]?.watcher.close).toHaveBeenCalledTimes(1);
    expect(watchCalls[1]?.watcher.close).toHaveBeenCalledTimes(1);

    index.close();
    fs.rmSync(tempDir, { recursive: true, force: true });
  });
});
