import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createIndexManager } from "../store.js";
import type { Chunk, MemorySource, StoredFile } from "../types.js";

const dimensions = 3;

const vectorAvailable = (() => {
  try {
    const db = new Database(":memory:");
    sqliteVec.load(db);
    db.close();
    return true;
  } catch {
    return false;
  }
})();

function makeFile(fileKey: string, source: MemorySource = "memory", contentHash = `hash:${fileKey}`): StoredFile {
  return {
    fileKey,
    contentHash,
    source,
    indexedAt: Date.now(),
  };
}

function makeChunk(
  text: string,
  startLine: number,
  endLine: number,
  headingContext?: string,
): Chunk {
  return {
    text,
    startLine,
    endLine,
    hash: `${startLine}:${endLine}:${text}`,
    headingContext,
  };
}

describe("createIndexManager", () => {
  let tempDir: string;
  let workspaceDir: string;
  let dbPath: string;
  let manager: ReturnType<typeof createIndexManager>;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-store-"));
    workspaceDir = path.join(tempDir, "workspace");
    fs.mkdirSync(workspaceDir, { recursive: true });
    dbPath = path.join(tempDir, "index.sqlite");
    manager = createIndexManager({
      dbPath,
      dimensions,
      workspaceDir,
    });
  });

  afterEach(() => {
    manager.close();
    fs.rmSync(tempDir, { recursive: true, force: true });
  });

  it("creates schema tables and triggers on init", () => {
    manager.close();

    const db = new Database(dbPath, { readonly: true });
    const objects = db
      .prepare<{ name: string }>(`
        SELECT name
        FROM sqlite_master
        WHERE name IN (
          'files',
          'chunks',
          'chunks_fts',
          'chunks_vec',
          'meta',
          'chunks_ai',
          'chunks_ad'
        );
      `)
      .all()
      .map((row) => row.name);

    expect(objects).toContain("files");
    expect(objects).toContain("chunks");
    expect(objects).toContain("chunks_fts");
    expect(objects).toContain("meta");
    expect(objects).toContain("chunks_ai");
    expect(objects).toContain("chunks_ad");
    if (vectorAvailable) {
      expect(objects).toContain("chunks_vec");
    }

    db.close();
  });

  it("indexes chunks and makes them retrievable", () => {
    const file = makeFile("memory/alpha.md");
    const chunks = [
      makeChunk("alpha token appears here", 1, 3, "Alpha"),
      makeChunk("beta token appears elsewhere", 4, 8, "Beta"),
    ];
    const vectors = [
      new Float32Array([1, 0, 0]),
      new Float32Array([0, 1, 0]),
    ];

    manager.indexFile(file, chunks, vectors);

    const results = manager.searchBM25("alpha", 10);
    expect(results.length).toBe(1);
    expect(results[0]?.fileKey).toBe(file.fileKey);
    expect(results[0]?.startLine).toBe(1);
    expect(results[0]?.headingContext).toBe("Alpha");
    expect(results[0]?.source).toBe("memory");
  });

  it("searches BM25 and returns normalized scores", () => {
    const file = makeFile("memory/search.md");
    manager.indexFile(
      file,
      [
        makeChunk("kiwi banana carrot", 1, 1),
        makeChunk("banana banana only", 2, 2),
      ],
      [new Float32Array([1, 0, 0]), new Float32Array([0, 1, 0])],
    );

    const results = manager.searchBM25("banana", 5);
    expect(results.length).toBeGreaterThan(0);
    expect(results.every((row) => row.score >= 0 && row.score <= 1)).toBe(true);
  });

  it("searchBM25 includes indexedAt from files table", () => {
    const indexedAt = 1_700_000_000_000;
    manager.indexFile(
      {
        ...makeFile("memory/indexed-bm25.md"),
        indexedAt,
      },
      [makeChunk("banana appears once", 1, 1)],
      [new Float32Array([1, 0, 0])],
    );

    const results = manager.searchBM25("banana", 5);
    expect(results).toHaveLength(1);
    expect(results[0]?.indexedAt).toBe(indexedAt);
  });

  it.skipIf(!vectorAvailable)("searches vectors and orders by distance", () => {
    const file = makeFile("memory/vector.md");
    manager.indexFile(
      file,
      [
        makeChunk("x-axis", 1, 1),
        makeChunk("y-axis", 2, 2),
        makeChunk("diagonal", 3, 3),
      ],
      [
        new Float32Array([1, 0, 0]),
        new Float32Array([0, 1, 0]),
        new Float32Array([0.5, 0.5, 0]),
      ],
    );

    const results = manager.searchVector(new Float32Array([1, 0, 0]), 3);
    expect(results.map((row) => row.text)).toEqual(["x-axis", "diagonal", "y-axis"]);
    expect(results[0]!.score).toBeGreaterThan(results[1]!.score);
    expect(results[1]!.score).toBeGreaterThan(results[2]!.score);
  });

  it.skipIf(!vectorAvailable)("searchVector includes indexedAt from files table", () => {
    const indexedAt = 1_710_000_000_000;
    manager.indexFile(
      {
        ...makeFile("memory/indexed-vector.md"),
        indexedAt,
      },
      [makeChunk("vector chunk", 1, 1)],
      [new Float32Array([1, 0, 0])],
    );

    const results = manager.searchVector(new Float32Array([1, 0, 0]), 1);
    expect(results).toHaveLength(1);
    expect(results[0]?.indexedAt).toBe(indexedAt);
  });

  it("removes file data from the index", () => {
    const file = makeFile("memory/remove.md");
    manager.indexFile(
      file,
      [makeChunk("remove me", 1, 1)],
      [new Float32Array([1, 0, 0])],
    );

    manager.removeFile(file.fileKey);

    const stats = manager.stats();
    expect(stats.files).toBe(0);
    expect(stats.chunks).toBe(0);
    expect(manager.searchBM25("remove", 5)).toEqual([]);
  });

  it("stores and returns file content hashes", () => {
    const file = makeFile("memory/hash.md", "memory", "abc123");
    manager.indexFile(
      file,
      [makeChunk("hash text", 1, 1)],
      [new Float32Array([1, 0, 0])],
    );

    expect(manager.getFileHash(file.fileKey)).toBe("abc123");
    expect(manager.getFileHash("missing.md")).toBeNull();
  });

  it("returns index stats including per-source counts", () => {
    manager.indexFile(
      makeFile("memory/one.md", "memory"),
      [makeChunk("m1", 1, 1), makeChunk("m2", 2, 2)],
      [new Float32Array([1, 0, 0]), new Float32Array([1, 0, 0])],
    );
    manager.indexFile(
      makeFile("sessions/two.md", "sessions"),
      [makeChunk("s1", 1, 1)],
      [new Float32Array([0, 1, 0])],
    );

    const stats = manager.stats();
    const sourceStats = new Map(stats.sources.map((row) => [row.source, row]));

    expect(stats.files).toBe(2);
    expect(stats.chunks).toBe(3);
    expect(stats.dbPath).toBe(dbPath);
    expect(stats.vectorDims).toBe(vectorAvailable ? dimensions : 0);
    expect(sourceStats.get("memory")?.files).toBe(1);
    expect(sourceStats.get("memory")?.chunks).toBe(2);
    expect(sourceStats.get("sessions")?.files).toBe(1);
    expect(sourceStats.get("sessions")?.chunks).toBe(1);
  });

  it("re-indexes an existing file key by replacing old chunks", () => {
    const fileKey = "memory/reindex.md";
    manager.indexFile(
      makeFile(fileKey, "memory", "old-hash"),
      [makeChunk("oldterm appears here", 1, 1)],
      [new Float32Array([1, 0, 0])],
    );

    manager.indexFile(
      makeFile(fileKey, "memory", "new-hash"),
      [makeChunk("newterm appears now", 10, 10)],
      [new Float32Array([0, 1, 0])],
    );

    expect(manager.getFileHash(fileKey)).toBe("new-hash");
    expect(manager.searchBM25("oldterm", 5)).toEqual([]);
    const newResults = manager.searchBM25("newterm", 5);
    expect(newResults.length).toBe(1);
    expect(newResults[0]?.startLine).toBe(10);
    expect(manager.stats().chunks).toBe(1);
  });

  it("returns empty results for malformed FTS5 queries", () => {
    manager.indexFile(
      makeFile("memory/fts.md"),
      [makeChunk("valid searchable text", 1, 1)],
      [new Float32Array([1, 0, 0])],
    );

    expect(manager.searchBM25("\"unterminated", 5)).toEqual([]);
  });

  it("normalizes readFileContent paths and returns relative paths", () => {
    const filePath = path.join(workspaceDir, "memory", "nested", "note.md");
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, "line1\nline2\nline3", "utf8");

    const withDotSlash = manager.readFileContent("./memory/nested/note.md");
    expect(withDotSlash).toEqual({
      text: "line1\nline2\nline3",
      path: "memory/nested/note.md",
    });

    const withBackslashes = manager.readFileContent(".\\memory\\nested\\note.md", 2, 1);
    expect(withBackslashes).toEqual({
      text: "line2",
      path: "memory/nested/note.md",
    });
  });
});
