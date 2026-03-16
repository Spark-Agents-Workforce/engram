import { DEFAULT_CONFIG, type EngramConfig, type MediaModality } from "./types.js";

const DEFAULT_MAX_MEDIA_FILE_BYTES = 10 * 1024 * 1024;
const DEFAULT_MEDIA_MODALITIES: MediaModality[] = ["image", "audio"];

function normalizeMediaModalities(modalities?: MediaModality[]): MediaModality[] {
  if (!Array.isArray(modalities) || modalities.length === 0) {
    return [...DEFAULT_MEDIA_MODALITIES];
  }

  const unique = new Set<MediaModality>();
  for (const modality of modalities) {
    if (modality === "image" || modality === "audio") {
      unique.add(modality);
    }
  }

  if (unique.size === 0) {
    return [...DEFAULT_MEDIA_MODALITIES];
  }

  return Array.from(unique);
}

export interface ResolvedConfig {
  geminiApiKey?: string;
  dimensions: 768 | 1536 | 3072;
  chunkTokens: number;
  chunkOverlap: number;
  reranking: boolean;
  timeDecay: {
    enabled: boolean;
    halfLifeDays: number;
  };
  maxSessionShare: number;
  multimodal: {
    enabled: boolean;
    modalities: MediaModality[];
    maxFileBytes: number;
  };
}

export function resolveConfig(raw?: EngramConfig | Record<string, unknown>): ResolvedConfig {
  const cfg = (raw ?? {}) as EngramConfig;

  return {
    geminiApiKey: cfg.geminiApiKey,
    dimensions: cfg.dimensions ?? DEFAULT_CONFIG.dimensions,
    chunkTokens: cfg.chunkTokens ?? DEFAULT_CONFIG.chunkTokens,
    chunkOverlap: cfg.chunkOverlap ?? DEFAULT_CONFIG.chunkOverlap,
    reranking: cfg.reranking ?? true,
    timeDecay: {
      enabled: cfg.timeDecay?.enabled ?? DEFAULT_CONFIG.timeDecay.enabled,
      halfLifeDays: cfg.timeDecay?.halfLifeDays ?? DEFAULT_CONFIG.timeDecay.halfLifeDays,
    },
    maxSessionShare: cfg.maxSessionShare ?? DEFAULT_CONFIG.maxSessionShare,
    multimodal: {
      enabled: cfg.multimodal?.enabled ?? true,
      modalities: normalizeMediaModalities(cfg.multimodal?.modalities),
      maxFileBytes: Math.max(1, Math.trunc(cfg.multimodal?.maxFileBytes ?? DEFAULT_MAX_MEDIA_FILE_BYTES)),
    },
  };
}
