import { describe, expect, it } from "vitest";
import { createLightweightReranker } from "../reranker.js";

describe("createLightweightReranker", () => {
  it("reorders candidates by query-term overlap", async () => {
    const reranker = createLightweightReranker();
    const results = await reranker.rerank("sqlite vector search", [
      { text: "meeting notes and roadmap", score: 0.9 },
      { text: "sqlite vector search with bm25 fusion", score: 0.4 },
      { text: "random unrelated text", score: 0.6 },
    ]);

    expect(results.map((row) => row.text)[0]).toBe("sqlite vector search with bm25 fusion");
  });

  it("returns [] for empty candidates", async () => {
    const reranker = createLightweightReranker();
    await expect(reranker.rerank("anything", [])).resolves.toEqual([]);
  });

  it("preserves all candidates", async () => {
    const reranker = createLightweightReranker();
    const input = [
      { text: "alpha beta", score: 0.5 },
      { text: "beta gamma", score: 0.4 },
      { text: "gamma delta", score: 0.3 },
    ];
    const output = await reranker.rerank("beta", input);

    expect(output).toHaveLength(input.length);
    expect(new Set(output.map((row) => row.text))).toEqual(new Set(input.map((row) => row.text)));
  });

  it("close() does not throw", async () => {
    const reranker = createLightweightReranker();
    await expect(reranker.close()).resolves.toBeUndefined();
  });
});
