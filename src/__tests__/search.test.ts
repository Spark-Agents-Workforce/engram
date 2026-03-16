import { describe, expect, it } from "vitest";
import {
  applySourceBalancing,
  applyTimeDecay,
  chunkKey,
  computeRRFScores,
  search,
} from "../search.js";
import type { Reranker } from "../reranker.js";
import type { EmbeddingClient, IndexManager, MemorySource, ScoredChunk } from "../types.js";

function makeChunk(
  fileKey: string,
  startLine: number,
  endLine: number,
  opts?: {
    text?: string;
    source?: MemorySource;
    score?: number;
    headingContext?: string;
    indexedAt?: number;
  },
): ScoredChunk {
  return {
    fileKey,
    startLine,
    endLine,
    text: opts?.text ?? `chunk ${fileKey}:${startLine}`,
    score: opts?.score ?? 0.5,
    source: opts?.source ?? "memory",
    headingContext: opts?.headingContext,
    indexedAt: opts?.indexedAt,
  };
}

function mockEmbedding(dims = 3): EmbeddingClient {
  return {
    dimensions: dims,
    model: "test-model",
    embedText: async () => new Float32Array(dims).fill(0.5),
    embedBatch: async (texts) => texts.map(() => new Float32Array(dims).fill(0.5)),
    supportsMultimodal: false,
  };
}

function mockIndex(bm25Results: ScoredChunk[], vectorResults: ScoredChunk[]): IndexManager {
  return {
    searchBM25: () => bm25Results,
    searchVector: () => vectorResults,
    indexFile: () => {},
    removeFile: () => {},
    getFileHash: () => null,
    readFileContent: () => null,
    stats: () => ({ files: 0, chunks: 0, sources: [], dbPath: ":memory:", vectorDims: 3 }),
    close: () => {},
  } as IndexManager;
}

describe("chunkKey", () => {
  it("produces fileKey:startLine:endLine format", () => {
    const key = chunkKey(makeChunk("notes/a.md", 10, 25));
    expect(key).toBe("notes/a.md:10:25");
  });
});

describe("computeRRFScores", () => {
  it("uses score = weight / (k + rank)", () => {
    const chunks = [
      makeChunk("a.md", 1, 1),
      makeChunk("b.md", 1, 1),
      makeChunk("c.md", 1, 1),
    ];

    const scores = computeRRFScores(chunks, 0.7, 60);

    expect(scores.get(chunkKey(chunks[0]))?.score).toBeCloseTo(0.7 / 61, 10);
    expect(scores.get(chunkKey(chunks[1]))?.score).toBeCloseTo(0.7 / 62, 10);
    expect(scores.get(chunkKey(chunks[2]))?.score).toBeCloseTo(0.7 / 63, 10);
  });

  it("returns empty map for empty input", () => {
    const scores = computeRRFScores([], 0.7, 60);
    expect(scores.size).toBe(0);
  });

  it("keeps first occurrence for duplicate chunk keys", () => {
    const first = makeChunk("dup.md", 1, 5);
    const duplicate = makeChunk("dup.md", 1, 5);
    const third = makeChunk("other.md", 1, 5);

    const scores = computeRRFScores([first, duplicate, third], 0.7, 60);

    expect(scores.size).toBe(2);
    expect(scores.get(chunkKey(first))?.score).toBeCloseTo(0.7 / 61, 10);
    expect(scores.get(chunkKey(third))?.score).toBeCloseTo(0.7 / 63, 10);
  });
});

describe("search", () => {
  it("gives co-occurring chunk a higher score than single-list chunks", async () => {
    const shared = makeChunk("shared.md", 1, 3);
    const bm25Only = makeChunk("bm25-only.md", 1, 3);
    const vectorOnly = makeChunk("vector-only.md", 1, 3);

    const results = await search(
      "query",
      mockEmbedding(),
      mockIndex([shared, bm25Only], [shared, vectorOnly]),
      { maxResults: 10 },
    );

    const sharedResult = results.find((r) => r.path === "shared.md");
    const bm25OnlyResult = results.find((r) => r.path === "bm25-only.md");
    const vectorOnlyResult = results.find((r) => r.path === "vector-only.md");

    expect(sharedResult).toBeDefined();
    expect(bm25OnlyResult).toBeDefined();
    expect(vectorOnlyResult).toBeDefined();
    expect(sharedResult!.score).toBeGreaterThan(bm25OnlyResult!.score);
    expect(sharedResult!.score).toBeGreaterThan(vectorOnlyResult!.score);
  });

  it("returns [] for empty query", async () => {
    const results = await search("", mockEmbedding(), mockIndex([], []), { maxResults: 5 });
    expect(results).toEqual([]);
  });

  it("returns BM25-only results when embedding fails", async () => {
    const embedding: EmbeddingClient = {
      dimensions: 3,
      model: "test-model",
      embedText: async () => {
        throw new Error("boom");
      },
      embedBatch: async (texts) => texts.map(() => new Float32Array(3).fill(0.5)),
      supportsMultimodal: false,
    };

    const bm25 = [makeChunk("bm25.md", 5, 8)];
    const results = await search("query", embedding, mockIndex(bm25, []), {
      maxResults: 5,
      bm25Weight: 0.3,
      rrfK: 60,
    });

    expect(results).toHaveLength(1);
    expect(results[0]?.path).toBe("bm25.md");
    expect(results[0]?.score).toBeCloseTo(0.3 / 61, 10);
  });

  it("returns vector-only results when BM25 is empty", async () => {
    const vector = [makeChunk("vector.md", 9, 10)];
    const results = await search("query", mockEmbedding(), mockIndex([], vector), {
      maxResults: 5,
      vectorWeight: 0.7,
      rrfK: 60,
    });

    expect(results).toHaveLength(1);
    expect(results[0]?.path).toBe("vector.md");
    expect(results[0]?.score).toBeCloseTo(0.7 / 61, 10);
  });

  it("returns [] when both BM25 and vector results are empty", async () => {
    const results = await search("query", mockEmbedding(), mockIndex([], []), { maxResults: 5 });
    expect(results).toEqual([]);
  });

  it("filters out rows below minScore", async () => {
    const vector = [
      makeChunk("v1.md", 1, 1),
      makeChunk("v2.md", 1, 1),
      makeChunk("v3.md", 1, 1),
    ];

    const results = await search("query", mockEmbedding(), mockIndex([], vector), {
      maxResults: 10,
      minScore: 0.5,
      vectorWeight: 1,
      bm25Weight: 0,
      rrfK: 0,
    });

    expect(results).toHaveLength(2);
    expect(results.every((r) => r.score >= 0.5)).toBe(true);
    expect(results.map((r) => r.path)).toEqual(["v1.md", "v2.md"]);
  });

  it("respects maxResults limit", async () => {
    const bm25 = [
      makeChunk("a.md", 1, 1),
      makeChunk("b.md", 1, 1),
      makeChunk("c.md", 1, 1),
    ];

    const results = await search("query", mockEmbedding(), mockIndex(bm25, []), {
      maxResults: 2,
    });

    expect(results).toHaveLength(2);
    expect(results.map((r) => r.path)).toEqual(["a.md", "b.md"]);
  });

  it("returns SearchResult with expected field mapping and citation format", async () => {
    const chunk = makeChunk("docs/file.ts", 1, 5, {
      text: "hello world",
      source: "sessions",
    });

    const results = await search("query", mockEmbedding(), mockIndex([chunk], []), {
      maxResults: 1,
    });

    expect(results).toHaveLength(1);
    expect(results[0]).toEqual({
      path: "docs/file.ts",
      startLine: 1,
      endLine: 5,
      score: 0.3 / 61,
      snippet: "hello world",
      source: "sessions",
      citation: "docs/file.ts#L1-L5",
    });
  });

  it("applies reranker ordering when provided", async () => {
    const early = makeChunk("early.md", 1, 1, { text: "general notes" });
    const later = makeChunk("later.md", 1, 1, { text: "sqlite vector search details" });

    const reranker: Reranker = {
      async rerank(_query, candidates) {
        return candidates
          .map((candidate) => ({
            ...candidate,
            rerankerScore: candidate.text.includes("sqlite") ? 1 : 0,
          }))
          .sort((a, b) => b.rerankerScore - a.rerankerScore);
      },
      async close() {},
    };

    const results = await search("sqlite search", mockEmbedding(), mockIndex([early, later], []), {
      maxResults: 10,
      reranker,
    });

    expect(results.map((row) => row.path)).toEqual(["later.md", "early.md"]);
  });

  it("keeps existing ordering when reranker is not provided", async () => {
    const first = makeChunk("first.md", 1, 1, { text: "general notes" });
    const second = makeChunk("second.md", 1, 1, { text: "sqlite vector search details" });

    const results = await search("sqlite search", mockEmbedding(), mockIndex([first, second], []), {
      maxResults: 10,
    });

    expect(results.map((row) => row.path)).toEqual(["first.md", "second.md"]);
  });
});

describe("applyTimeDecay", () => {
  it("reduces older scores based on chunk.indexedAt", () => {
    const now = Date.now();

    const fresh = makeChunk("memory/fresh.md", 1, 1, { indexedAt: now });
    const old = makeChunk("memory/old.md", 1, 1, { indexedAt: now - 2 * 24 * 60 * 60 * 1000 });
    const missingTimestamp = makeChunk("memory/no-indexed-at.md", 1, 1);

    const results = [
      { chunk: fresh, score: 1 },
      { chunk: old, score: 1 },
      { chunk: missingTimestamp, score: 1 },
    ];

    applyTimeDecay(results, 1);

    expect(results[0].score).toBeCloseTo(1, 6);
    expect(results[1].score).toBeCloseTo(0.25, 6);
    expect(results[1].score).toBeLessThan(results[0].score);
    expect(results[2].score).toBe(1);
  });
});

describe("applySourceBalancing", () => {
  it("caps session results to maxSessionShare of maxResults", () => {
    const sixSessions = Array.from({ length: 6 }, (_, i) => ({
      chunk: makeChunk(`s${i + 1}.md`, 1, 1, { source: "sessions" }),
      score: 1 - i * 0.01,
    }));
    const fourMemory = Array.from({ length: 4 }, (_, i) => ({
      chunk: makeChunk(`m${i + 1}.md`, 1, 1, { source: "memory" }),
      score: 0.8 - i * 0.01,
    }));

    const balanced = applySourceBalancing([...sixSessions, ...fourMemory], 0.4, 10);
    const sessionCount = balanced.filter((r) => r.chunk.source === "sessions").length;

    expect(sessionCount).toBeLessThanOrEqual(4);
    expect(sessionCount).toBe(4);
    expect(balanced).toHaveLength(8);
  });

  it("keeps all rows when maxSessionShare is 1.0", () => {
    const rows = [
      { chunk: makeChunk("s1.md", 1, 1, { source: "sessions" }), score: 1 },
      { chunk: makeChunk("s2.md", 1, 1, { source: "sessions" }), score: 0.9 },
      { chunk: makeChunk("m1.md", 1, 1, { source: "memory" }), score: 0.8 },
    ];

    const balanced = applySourceBalancing(rows, 1.0, 10);
    expect(balanced).toEqual(rows);
  });

  it("returns empty array for empty input", () => {
    expect(applySourceBalancing([], 0.4, 10)).toEqual([]);
  });
});
