import type { EmbeddingClient, GeminiTaskType } from "./types.js";

const GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com";
const GEMINI_DEFAULT_MODEL = "gemini-embedding-2-preview";
const DEFAULT_DIMENSIONS = 768;
const MAX_BATCH_SIZE = 100;

type JsonObject = Record<string, unknown>;

function isRecord(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null;
}

function normalizeModelName(model: string): string {
  return model.startsWith("models/") ? model.slice("models/".length) : model;
}

function modelResourceName(model: string): string {
  return model.startsWith("models/") ? model : `models/${model}`;
}

function getNumberArray(value: unknown, context: string): number[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array`);
  }

  const out = new Array<number>(value.length);
  for (let i = 0; i < value.length; i++) {
    const item = value[i];
    if (typeof item !== "number") {
      throw new Error(`${context} contains a non-number value at index ${i}`);
    }
    out[i] = item;
  }
  return out;
}

function toGeminiVector(values: number[], dimensions: number): Float32Array {
  const vec = Float32Array.from(values);
  if (dimensions < 3072) {
    return l2Normalize(vec);
  }
  return vec;
}

function buildGeminiEndpoint(params: {
  baseUrl: string;
  model: string;
  method: "embedContent" | "batchEmbedContents";
  apiKey: string;
}): string {
  const root = params.baseUrl.replace(/\/+$/, "");
  return `${root}/v1beta/models/${params.model}:${params.method}?key=${encodeURIComponent(params.apiKey)}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function postJsonWithRetry(params: {
  url: string;
  body: unknown;
  providerName: string;
  headers?: Record<string, string>;
}): Promise<unknown> {
  for (let attempt = 0; attempt < 2; attempt++) {
    const response = await fetch(params.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(params.headers ?? {}),
      },
      body: JSON.stringify(params.body),
    });

    if (response.status === 429 && attempt === 0) {
      await sleep(1000);
      continue;
    }

    if (!response.ok) {
      const responseBody = await response.text();
      throw new Error(
        `${params.providerName} embedding request failed (${response.status}): ${responseBody || "<empty response body>"}`
      );
    }

    return (await response.json()) as unknown;
  }

  throw new Error(`${params.providerName} embedding request failed after retry`);
}

function parseGeminiEmbeddingResponse(response: unknown, dimensions: number): Float32Array {
  if (!isRecord(response)) {
    throw new Error("Gemini embedding response must be an object");
  }

  const embedding = response.embedding;
  if (!isRecord(embedding)) {
    throw new Error("Gemini embedding response missing embedding object");
  }

  return toGeminiVector(getNumberArray(embedding.values, "embedding.values"), dimensions);
}

export function l2Normalize(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / norm;
  return out;
}

export async function createEmbeddingClient(params: {
  apiKey: string;
  dimensions?: number;
  model?: string;
  baseUrl?: string;
}): Promise<EmbeddingClient> {
  const model = normalizeModelName(params.model ?? GEMINI_DEFAULT_MODEL);
  const dimensions = params.dimensions ?? DEFAULT_DIMENSIONS;
  const baseUrl = params.baseUrl ?? GEMINI_DEFAULT_BASE_URL;

  const embedText = async (text: string, taskType: GeminiTaskType): Promise<Float32Array> => {
    const endpoint = buildGeminiEndpoint({
      baseUrl,
      model,
      method: "embedContent",
      apiKey: params.apiKey,
    });

    const body = {
      content: {
        parts: [{ text }],
      },
      taskType,
      outputDimensionality: dimensions,
    };

    const response = await postJsonWithRetry({
      url: endpoint,
      body,
      providerName: "Gemini",
    });
    return parseGeminiEmbeddingResponse(response, dimensions);
  };

  const embedMedia = async (data: Buffer, mimeType: string): Promise<Float32Array> => {
    const endpoint = buildGeminiEndpoint({
      baseUrl,
      model,
      method: "embedContent",
      apiKey: params.apiKey,
    });

    const body = {
      content: {
        parts: [{ inlineData: { data: data.toString("base64"), mimeType } }],
      },
      taskType: "RETRIEVAL_DOCUMENT" as const,
      outputDimensionality: dimensions,
    };

    const response = await postJsonWithRetry({
      url: endpoint,
      body,
      providerName: "Gemini",
    });
    return parseGeminiEmbeddingResponse(response, dimensions);
  };

  const embedBatch = async (texts: string[], taskType: GeminiTaskType): Promise<Float32Array[]> => {
    if (texts.length === 0) {
      return [];
    }

    const endpoint = buildGeminiEndpoint({
      baseUrl,
      model,
      method: "batchEmbedContents",
      apiKey: params.apiKey,
    });

    const vectors: Float32Array[] = [];
    for (let i = 0; i < texts.length; i += MAX_BATCH_SIZE) {
      const chunk = texts.slice(i, i + MAX_BATCH_SIZE);
      const body = {
        requests: chunk.map((text) => ({
          model: modelResourceName(model),
          content: {
            parts: [{ text }],
          },
          taskType,
          outputDimensionality: dimensions,
        })),
      };

      const response = await postJsonWithRetry({
        url: endpoint,
        body,
        providerName: "Gemini",
      });
      if (!isRecord(response) || !Array.isArray(response.embeddings)) {
        throw new Error("Gemini batch embedding response missing embeddings array");
      }

      if (response.embeddings.length !== chunk.length) {
        throw new Error(
          `Gemini batch embedding response count mismatch: expected ${chunk.length}, got ${response.embeddings.length}`
        );
      }

      for (let j = 0; j < response.embeddings.length; j++) {
        const embedding = response.embeddings[j];
        if (!isRecord(embedding)) {
          throw new Error(`Gemini batch embedding at index ${j} is not an object`);
        }
        vectors.push(toGeminiVector(getNumberArray(embedding.values, `embeddings[${j}].values`), dimensions));
      }
    }

    return vectors;
  };

  return {
    embedText,
    embedBatch,
    embedMedia,
    supportsMultimodal: true,
    dimensions,
    model,
  };
}
