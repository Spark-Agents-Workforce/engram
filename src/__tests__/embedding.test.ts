import { afterEach, describe, expect, it, vi } from "vitest";
import { createEmbeddingClient, l2Normalize } from "../embedding.js";

function createJsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as unknown as Response;
}

function createErrorResponse(status: number, body: string): Response {
  return {
    ok: false,
    status,
    json: async () => ({ error: body }),
    text: async () => body,
  } as unknown as Response;
}

describe("l2Normalize", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("normalizes known values", () => {
    const result = l2Normalize(new Float32Array([3, 4]));
    expect(result[0]).toBeCloseTo(0.6);
    expect(result[1]).toBeCloseTo(0.8);
  });

  it("preserves zero vectors", () => {
    const input = new Float32Array([0, 0, 0]);
    const result = l2Normalize(input);
    expect(result).toBe(input);
    expect(Array.from(result)).toEqual([0, 0, 0]);
  });

  it("keeps unit vectors unchanged", () => {
    const result = l2Normalize(new Float32Array([1, 0, 0]));
    expect(Array.from(result)).toEqual([1, 0, 0]);
  });
});

describe("createEmbeddingClient", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("sets model and dimensions", async () => {
    const defaultClient = await createEmbeddingClient({ apiKey: "key" });
    expect(defaultClient.model).toBe("gemini-embedding-2-preview");
    expect(defaultClient.dimensions).toBe(768);
    expect(defaultClient.supportsMultimodal).toBe(true);

    const client = await createEmbeddingClient({
      apiKey: "key",
      model: "custom-model",
      dimensions: 1536,
    });

    expect(client.model).toBe("custom-model");
    expect(client.dimensions).toBe(1536);
    expect(client.supportsMultimodal).toBe(true);
  });

  it("embeds text with correct request format", async () => {
    const fetchMock = vi.fn(async () => createJsonResponse({ embedding: { values: [3, 4] } }));
    vi.stubGlobal("fetch", fetchMock);

    const client = await createEmbeddingClient({ apiKey: "test-key" });
    const vector = await client.embedText("hello", "RETRIEVAL_DOCUMENT");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent?key=test-key"
    );
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });

    const body = JSON.parse(String(init.body));
    expect(body).toEqual({
      content: { parts: [{ text: "hello" }] },
      taskType: "RETRIEVAL_DOCUMENT",
      outputDimensionality: 768,
    });

    expect(vector[0]).toBeCloseTo(0.6);
    expect(vector[1]).toBeCloseTo(0.8);
  });

  it("embeds media with inline base64 data", async () => {
    const fetchMock = vi.fn(async () => createJsonResponse({ embedding: { values: [3, 4] } }));
    vi.stubGlobal("fetch", fetchMock);

    const client = await createEmbeddingClient({ apiKey: "test-key" });
    const media = Buffer.from([0, 1, 2, 3, 4]);
    const vector = await client.embedMedia!(media, "image/png");

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent?key=test-key"
    );

    const body = JSON.parse(String(init.body));
    expect(body).toEqual({
      content: {
        parts: [{ inlineData: { data: media.toString("base64"), mimeType: "image/png" } }],
      },
      taskType: "RETRIEVAL_DOCUMENT",
      outputDimensionality: 768,
    });

    expect(vector[0]).toBeCloseTo(0.6);
    expect(vector[1]).toBeCloseTo(0.8);
  });

  it("splits batch requests larger than 100 items", async () => {
    const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body)) as {
        requests: Array<{ content: { parts: Array<{ text: string }> } }>;
      };

      return createJsonResponse({
        embeddings: body.requests.map(() => ({ values: [1, 0] })),
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await createEmbeddingClient({ apiKey: "test-key" });
    const texts = Array.from({ length: 205 }, (_, i) => `text-${i}`);
    const vectors = await client.embedBatch(texts, "RETRIEVAL_QUERY");

    expect(vectors).toHaveLength(205);
    expect(fetchMock).toHaveBeenCalledTimes(3);

    const firstCall = fetchMock.mock.calls[0] as [string, RequestInit];
    const secondCall = fetchMock.mock.calls[1] as [string, RequestInit];
    const thirdCall = fetchMock.mock.calls[2] as [string, RequestInit];

    expect(firstCall[0]).toContain(":batchEmbedContents?key=test-key");
    expect(secondCall[0]).toContain(":batchEmbedContents?key=test-key");
    expect(thirdCall[0]).toContain(":batchEmbedContents?key=test-key");

    const firstBody = JSON.parse(String(firstCall[1].body));
    const secondBody = JSON.parse(String(secondCall[1].body));
    const thirdBody = JSON.parse(String(thirdCall[1].body));

    expect(firstBody.requests).toHaveLength(100);
    expect(secondBody.requests).toHaveLength(100);
    expect(thirdBody.requests).toHaveLength(5);
    expect(firstBody.requests[0].taskType).toBe("RETRIEVAL_QUERY");
    expect(firstBody.requests[0].outputDimensionality).toBe(768);
    expect(firstBody.requests[0].model).toBe("models/gemini-embedding-2-preview");
  });

  it("retries once on 429 responses", async () => {
    vi.useFakeTimers();

    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(createErrorResponse(429, "rate limited"))
      .mockResolvedValueOnce(createJsonResponse({ embedding: { values: [1, 0] } }));
    vi.stubGlobal("fetch", fetchMock);

    const client = await createEmbeddingClient({ apiKey: "test-key" });
    const pending = client.embedText("hello", "RETRIEVAL_QUERY");
    await vi.runAllTimersAsync();
    const result = await pending;

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(Array.from(result)).toEqual([1, 0]);
  });

  it("throws descriptive errors on non-429 failures", async () => {
    const fetchMock = vi.fn().mockResolvedValue(createErrorResponse(400, "bad request payload"));
    vi.stubGlobal("fetch", fetchMock);

    const client = await createEmbeddingClient({ apiKey: "test-key" });

    await expect(client.embedText("hello", "RETRIEVAL_DOCUMENT")).rejects.toThrow(
      /Gemini embedding request failed \(400\): bad request payload/
    );
  });
});
