import type { ResolvedConfig } from "./config.js";
import type { Reranker } from "./reranker.js";
import { search } from "./search.js";
import type { SyncManager } from "./sync.js";
import type { EmbeddingClient, IndexManager, SearchResult, SyncOptions } from "./types.js";

const DEFAULT_RRF_K = 60;
const DEFAULT_VECTOR_WEIGHT = 0.7;
const DEFAULT_BM25_WEIGHT = 0.3;

export interface EngramManager {
  /** Search memory (delegates to search.ts) */
  search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      sessionKey?: string;
    },
  ): Promise<SearchResult[]>;

  /** Read a file by relative path with optional line range */
  readFile(params: {
    relPath: string;
    from?: number;
    lines?: number;
  }): Promise<{ text: string; path: string }>;

  /** Get provider status */
  status(): EngramStatus;

  /** Sync the index */
  sync(opts?: SyncOptions): Promise<void>;

  /** Lazy sync before search if dirty */
  syncIfDirty(): void;

  /** Sync once for a session key */
  warmSession(sessionKey?: string): Promise<void>;

  /** Test if embedding API is reachable */
  probeEmbeddingAvailability(): Promise<{ ok: boolean; error?: string }>;

  /** Test if sqlite-vec is loaded */
  probeVectorAvailability(): Promise<boolean>;

  /** Clean up resources */
  close(): Promise<void>;
}

export interface EngramStatus {
  backend: string;
  provider: string;
  model: string;
  files: number;
  chunks: number;
  dirty: boolean;
  workspaceDir: string;
  dbPath: string;
  sources: Array<{ source: "memory" | "sessions"; files: number; chunks: number }>;
  vector: {
    enabled: boolean;
    dims: number;
  };
  fts: {
    enabled: boolean;
    available: boolean;
  };
  custom: {
    rrfK: number;
    vectorWeight: number;
    bm25Weight: number;
    chunkTokens: number;
    chunkOverlap: number;
  };
}

function inferProvider(model: string): string {
  const normalized = model.toLowerCase();
  if (normalized.includes("gemini")) {
    return "gemini";
  }
  if (normalized.includes("text-embedding")) {
    return "openai";
  }
  return "gemini";
}

export function createEngramManager(params: {
  index: IndexManager;
  embedding: EmbeddingClient;
  syncManager: SyncManager;
  reranker?: Reranker | null;
  config: ResolvedConfig;
  workspaceDir: string;
}): EngramManager {
  const { index, embedding, syncManager, reranker, config, workspaceDir } = params;

  return {
    async search(
      query: string,
      opts?: {
        maxResults?: number;
        minScore?: number;
        sessionKey?: string;
      },
    ): Promise<SearchResult[]> {
      return search(query, embedding, index, {
        maxResults: opts?.maxResults,
        minScore: opts?.minScore,
        sessionKey: opts?.sessionKey,
        vectorWeight: DEFAULT_VECTOR_WEIGHT,
        bm25Weight: DEFAULT_BM25_WEIGHT,
        rrfK: DEFAULT_RRF_K,
        timeDecay: config.timeDecay,
        maxSessionShare: config.maxSessionShare,
        reranker: reranker ?? null,
      });
    },

    async readFile(params: { relPath: string; from?: number; lines?: number }): Promise<{ text: string; path: string }> {
      const result = index.readFileContent(params.relPath, params.from, params.lines);
      if (result) {
        return result;
      }
      return { text: "", path: params.relPath };
    },

    status(): EngramStatus {
      const stats = index.stats();
      const vectorEnabled = stats.vectorDims > 0;
      return {
        backend: "engram",
        provider: inferProvider(embedding.model),
        model: embedding.model,
        files: stats.files,
        chunks: stats.chunks,
        dirty: syncManager.isDirty(),
        workspaceDir,
        dbPath: stats.dbPath,
        sources: stats.sources,
        vector: {
          enabled: vectorEnabled,
          dims: stats.vectorDims,
        },
        fts: {
          enabled: true,
          available: true,
        },
        custom: {
          rrfK: DEFAULT_RRF_K,
          vectorWeight: DEFAULT_VECTOR_WEIGHT,
          bm25Weight: DEFAULT_BM25_WEIGHT,
          chunkTokens: config.chunkTokens,
          chunkOverlap: config.chunkOverlap,
        },
      };
    },

    async sync(opts?: SyncOptions): Promise<void> {
      await syncManager.sync(opts);
    },

    syncIfDirty(): void {
      syncManager.syncIfDirty();
    },

    async warmSession(sessionKey?: string): Promise<void> {
      await syncManager.warmSession(sessionKey);
    },

    async probeEmbeddingAvailability(): Promise<{ ok: boolean; error?: string }> {
      try {
        await embedding.embedText("test", "RETRIEVAL_QUERY");
        return { ok: true };
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { ok: false, error: message };
      }
    },

    async probeVectorAvailability(): Promise<boolean> {
      return index.stats().vectorDims > 0;
    },

    async close(): Promise<void> {
      if (reranker) {
        try {
          await reranker.close();
        } catch {
          // Keep shutdown best-effort.
        }
      }
      index.close();
      syncManager.close();
    },
  };
}
