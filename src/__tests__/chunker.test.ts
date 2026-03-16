import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { chunkMarkdown, estimateTokens } from "../chunker.js";

describe("chunkMarkdown", () => {
  it("returns [] for empty content", () => {
    expect(chunkMarkdown("", { maxTokens: 100, overlapRatio: 0.15 })).toEqual([]);
  });

  it("returns a single chunk for short content with correct lines", () => {
    const content = "Hello world\nSecond line";
    const chunks = chunkMarkdown(content, { maxTokens: 100, overlapRatio: 0 });

    expect(chunks).toHaveLength(1);
    expect(chunks[0]?.text).toBe(content);
    expect(chunks[0]?.startLine).toBe(1);
    expect(chunks[0]?.endLine).toBe(2);
  });

  it("splits at heading boundaries when possible", () => {
    const content = [
      "# A",
      "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha",
      "## B",
      "beta beta beta beta beta beta beta beta beta beta beta beta beta beta beta beta",
    ].join("\n");

    const chunks = chunkMarkdown(content, { maxTokens: 20, overlapRatio: 0 });

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.some((chunk) => chunk.startLine === 3 && chunk.text.startsWith("## B"))).toBe(true);
  });

  it("never splits in the middle of fenced code blocks", () => {
    const codeLines = Array.from({ length: 40 }, (_, i) => `const v${i} = ${i};`).join("\n");
    const content = `Intro.\n\n\`\`\`ts\n${codeLines}\n\`\`\`\n\nTail.`;

    const chunks = chunkMarkdown(content, { maxTokens: 40, overlapRatio: 0 });

    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      const fenceCount = (chunk.text.match(/```/g) ?? []).length;
      expect(fenceCount === 0 || fenceCount % 2 === 0).toBe(true);
    }
    expect(chunks.some((chunk) => estimateTokens(chunk.text) > 40)).toBe(true);
  });

  it("strips frontmatter from chunks", () => {
    const content = ["---", "title: Demo", "tags: [a, b]", "---", "# Header", "Body text."].join("\n");
    const chunks = chunkMarkdown(content, { maxTokens: 100, overlapRatio: 0 });

    expect(chunks).toHaveLength(1);
    expect(chunks[0]?.text).toContain("# Header");
    expect(chunks[0]?.text).not.toContain("title: Demo");
    expect(chunks[0]?.startLine).toBe(5);
  });

  it("applies overlap between chunks", () => {
    const content = [
      "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six.",
    ].join("\n");
    const chunks = chunkMarkdown(content, { maxTokens: 12, overlapRatio: 0.25 });

    expect(chunks.length).toBeGreaterThan(1);

    const a = chunks[0]?.text ?? "";
    const b = chunks[1]?.text ?? "";
    let overlap = 0;
    const max = Math.min(a.length, b.length);
    for (let i = max; i >= 1; i -= 1) {
      if (a.slice(-i) === b.slice(0, i)) {
        overlap = i;
        break;
      }
    }

    expect(overlap).toBeGreaterThan(0);
  });

  it("tracks parent heading hierarchy in headingContext", () => {
    const content = [
      "# Config",
      "General setup details that are long enough to encourage chunking.",
      "## Auth",
      "Authentication details that are also long enough to split.",
      "### Keys",
      "Key handling details continue here.",
    ].join("\n");

    const chunks = chunkMarkdown(content, { maxTokens: 14, overlapRatio: 0 });

    const authChunk = chunks.find((chunk) => chunk.text.includes("Authentication details"));
    const keysChunk = chunks.find((chunk) => chunk.text.includes("Key handling details"));

    expect(authChunk?.headingContext).toBe("# Config > ## Auth");
    expect(keysChunk?.headingContext).toBe("# Config > ## Auth > ### Keys");
  });

  it("reports 1-indexed startLine and endLine correctly", () => {
    const content = ["# A", "A body line", "## B", "B body line", "## C", "C body line"].join("\n");
    const chunks = chunkMarkdown(content, { maxTokens: 8, overlapRatio: 0 });

    expect(chunks.length).toBeGreaterThanOrEqual(3);
    expect(chunks[0]?.startLine).toBe(1);
    expect(chunks[0]?.endLine).toBe(2);
    expect(chunks[1]?.startLine).toBe(3);
    expect(chunks[1]?.endLine).toBe(4);
    expect(chunks[2]?.startLine).toBe(5);
    expect(chunks[2]?.endLine).toBe(6);
  });

  it("generates deterministic 16-char hashes", () => {
    const content = "# Title\nSome text for hashing.";
    const chunksA = chunkMarkdown(content, { maxTokens: 100, overlapRatio: 0 });
    const chunksB = chunkMarkdown(content, { maxTokens: 100, overlapRatio: 0 });

    expect(chunksA).toHaveLength(1);
    expect(chunksA[0]?.hash).toBe(chunksB[0]?.hash);
    expect(chunksA[0]?.hash).toHaveLength(16);

    const expected = createHash("sha256").update(chunksA[0]?.text ?? "").digest("hex").slice(0, 16);
    expect(chunksA[0]?.hash).toBe(expected);
  });

  it("chunks large content into multiple roughly max-sized chunks", () => {
    const sentence = "This is a long sentence for chunking behavior validation. ";
    const content = sentence.repeat(200);
    const maxTokens = 40;
    const chunks = chunkMarkdown(content, { maxTokens, overlapRatio: 0 });

    expect(chunks.length).toBeGreaterThan(5);
    for (const chunk of chunks) {
      expect(chunk.text.length).toBeGreaterThan(0);
      expect(estimateTokens(chunk.text)).toBeLessThanOrEqual(maxTokens + 1);
    }
  });
});
