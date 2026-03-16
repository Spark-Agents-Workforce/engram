import { describe, expect, it } from "vitest";
import { resolveConfig } from "../config.js";

describe("resolveConfig", () => {
  it("applies defaults", () => {
    const config = resolveConfig({});

    expect(config.dimensions).toBe(768);
    expect(config.chunkTokens).toBe(1024);
    expect(config.chunkOverlap).toBe(0.15);
    expect(config.maxSessionShare).toBe(0.4);
    expect(config.timeDecay).toEqual({
      enabled: true,
      halfLifeDays: 30,
    });
    expect(config.reranking).toBe(true);
    expect(config.multimodal).toEqual({
      enabled: false,
      modalities: ["image", "audio"],
      maxFileBytes: 10 * 1024 * 1024,
    });
  });

  it("applies overrides", () => {
    const config = resolveConfig({
      dimensions: 3072,
      chunkTokens: 2048,
      chunkOverlap: 0.2,
      reranking: false,
      timeDecay: {
        enabled: false,
        halfLifeDays: 14,
      },
      maxSessionShare: 0.75,
      geminiApiKey: "abc123",
      multimodal: {
        enabled: true,
        modalities: ["image", "audio"],
        maxFileBytes: 512 * 1024,
      },
    });

    expect(config.dimensions).toBe(3072);
    expect(config.chunkTokens).toBe(2048);
    expect(config.chunkOverlap).toBe(0.2);
    expect(config.reranking).toBe(false);
    expect(config.timeDecay).toEqual({
      enabled: false,
      halfLifeDays: 14,
    });
    expect(config.maxSessionShare).toBe(0.75);
    expect(config.geminiApiKey).toBe("abc123");
    expect(config.multimodal).toEqual({
      enabled: true,
      modalities: ["image", "audio"],
      maxFileBytes: 512 * 1024,
    });
  });

  it("handles undefined input", () => {
    const config = resolveConfig(undefined);

    expect(config.dimensions).toBe(768);
    expect(config.chunkTokens).toBe(1024);
    expect(config.chunkOverlap).toBe(0.15);
    expect(config.maxSessionShare).toBe(0.4);
    expect(config.timeDecay).toEqual({
      enabled: true,
      halfLifeDays: 30,
    });
    expect(config.multimodal).toEqual({
      enabled: false,
      modalities: ["image", "audio"],
      maxFileBytes: 10 * 1024 * 1024,
    });
  });
});
