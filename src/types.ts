export type MediaModality = "image" | "audio";

export const MEDIA_EXTENSIONS: Record<MediaModality, string[]> = {
  image: [".jpg", ".jpeg", ".png", ".webp", ".gif"],
  audio: [".mp3", ".wav", ".ogg", ".opus", ".m4a", ".aac", ".flac"],
};

export const MEDIA_MIME_TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".mp3": "audio/mpeg",
  ".wav": "audio/wav",
  ".ogg": "audio/ogg",
  ".opus": "audio/opus",
  ".m4a": "audio/mp4",
  ".aac": "audio/aac",
  ".flac": "audio/flac",
};

export interface EngramConfig {
  geminiApiKey?: string;
  dimensions?: 768 | 1536 | 3072;
  chunkTokens?: number;
  chunkOverlap?: number;
  reranking?: boolean;
  timeDecay?: {
    enabled?: boolean;
    halfLifeDays?: number;
  };
  maxSessionShare?: number;
  multimodal?: {
    enabled?: boolean;
    modalities?: MediaModality[];
    maxFileBytes?: number;
  };
}

export const DEFAULT_CONFIG: Required<
  Pick<EngramConfig, "dimensions" | "chunkTokens" | "chunkOverlap" | "maxSessionShare">
> & {
  timeDecay: Required<NonNullable<EngramConfig["timeDecay"]>>;
} = {
  dimensions: 768,
  chunkTokens: 1024,
  chunkOverlap: 0.15,
  maxSessionShare: 0.4,
  timeDecay: {
    enabled: true,
    halfLifeDays: 30,
  },
};

export type GeminiTaskType = "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY";

export interface EmbeddingClient {
  embedText(text: string, taskType: GeminiTaskType): Promise<Float32Array>;
  embedBatch(texts: string[], taskType: GeminiTaskType): Promise<Float32Array[]>;
  /** Embed a binary file (image, audio, PDF). Returns a normalized Float32Array. */
  embedMedia?(data: Buffer, mimeType: string): Promise<Float32Array>;
  /** Whether this client supports multimodal embedding. */
  readonly supportsMultimodal: boolean;
  dimensions: number;
  model: string;
}

export interface Chunk {
  text: string;
  startLine: number;
  endLine: number;
  hash: string;
  headingContext?: string;
}

export interface ChunkerOptions {
  maxTokens: number;
  overlapRatio: number;
}

export type MemorySource = "memory" | "sessions";

export interface StoredFile {
  fileKey: string;
  contentHash: string;
  source: MemorySource;
  indexedAt: number;
}

export interface ScoredChunk {
  fileKey: string;
  startLine: number;
  endLine: number;
  text: string;
  score: number;
  source: MemorySource;
  headingContext?: string;
  indexedAt?: number;
}

export interface IndexStats {
  files: number;
  chunks: number;
  sources: Array<{ source: MemorySource; files: number; chunks: number }>;
  dbPath: string;
  vectorDims: number;
}

export interface IndexManager {
  indexFile(file: StoredFile, chunks: Chunk[], vectors: Float32Array[]): void;
  removeFile(fileKey: string): void;
  searchBM25(query: string, topK: number): ScoredChunk[];
  searchVector(queryVec: Float32Array, topK: number): ScoredChunk[];
  getFileHash(fileKey: string): string | null;
  readFileContent(relPath: string, from?: number, lines?: number): { text: string; path: string } | null;
  stats(): IndexStats;
  close(): void;
}

export interface SearchOptions {
  maxResults?: number;
  minScore?: number;
  sessionKey?: string;
}

export interface SearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: MemorySource;
  citation?: string;
}

export interface SyncProgress {
  completed: number;
  total: number;
  label?: string;
}

export interface SyncOptions {
  reason?: string;
  force?: boolean;
  sessionFiles?: string[];
  progress?: (update: SyncProgress) => void;
}
