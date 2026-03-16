import type {
  EmbeddingClient,
  IndexManager,
  ScoredChunk,
  SearchOptions,
  SearchResult,
} from "./types.js";
import type { Reranker } from "./reranker.js";

const DEFAULT_VECTOR_WEIGHT = 0.7;
const DEFAULT_BM25_WEIGHT = 0.3;
const DEFAULT_RRF_K = 60;
const DEFAULT_MAX_RESULTS = 10;
const DEFAULT_MIN_SCORE = 0.0;
const CO_OCCURRENCE_BONUS = 0.05;
const CANDIDATE_TOP_K = 50;
const RERANK_TOP_K = 20;

interface RankedChunk {
  chunk: ScoredChunk;
  score: number;
}

interface MergedRankedChunk extends RankedChunk {
  inVector: boolean;
  inBM25: boolean;
}

function getChunkTimestampMs(chunk: ScoredChunk): number | null {
  return chunk.indexedAt ?? null;
}

/** Create a unique key for a chunk (for deduplication/merging) */
export function chunkKey(chunk: ScoredChunk): string {
  return `${chunk.fileKey}:${chunk.startLine}:${chunk.endLine}`;
}

/** Compute RRF scores for a ranked list of chunks */
export function computeRRFScores(
  chunks: ScoredChunk[],
  weight: number,
  k: number,
): Map<string, { chunk: ScoredChunk; score: number }> {
  const scores = new Map<string, { chunk: ScoredChunk; score: number }>();
  if (chunks.length === 0 || weight <= 0) {
    return scores;
  }

  const safeK = Number.isFinite(k) && k >= 0 ? k : DEFAULT_RRF_K;
  for (let i = 0; i < chunks.length; i += 1) {
    const chunk = chunks[i];
    const key = chunkKey(chunk);
    if (scores.has(key)) {
      continue;
    }

    const rank = i + 1;
    scores.set(key, {
      chunk,
      score: weight / (safeK + rank),
    });
  }

  return scores;
}

/** Apply exponential time decay to scores */
export function applyTimeDecay(
  results: Array<{ chunk: ScoredChunk; score: number }>,
  halfLifeDays: number,
): void {
  if (halfLifeDays <= 0 || !Number.isFinite(halfLifeDays)) {
    return;
  }

  const nowMs = Date.now();
  const msPerDay = 24 * 60 * 60 * 1000;

  for (const result of results) {
    const timestampMs = getChunkTimestampMs(result.chunk);
    if (timestampMs === null) {
      continue;
    }

    const ageDays = Math.max(0, (nowMs - timestampMs) / msPerDay);
    const decayMultiplier = Math.pow(0.5, ageDays / halfLifeDays);
    result.score *= decayMultiplier;
  }
}

/** Apply source balancing — demote excess session results */
export function applySourceBalancing(
  results: Array<{ chunk: ScoredChunk; score: number }>,
  maxSessionShare: number,
  maxResults: number,
): Array<{ chunk: ScoredChunk; score: number }> {
  if (results.length === 0 || maxResults <= 0) {
    return [];
  }

  if (!Number.isFinite(maxSessionShare) || maxSessionShare >= 1) {
    return results.slice();
  }

  const sessionCap = Math.max(0, Math.floor(maxSessionShare * maxResults));
  const balanced: Array<{ chunk: ScoredChunk; score: number }> = [];
  let sessionCount = 0;

  for (const result of results) {
    if (result.chunk.source === "sessions") {
      if (sessionCount < sessionCap) {
        balanced.push(result);
        sessionCount += 1;
      }
      continue;
    }

    balanced.push(result);
  }

  return balanced;
}

export async function search(
  query: string,
  embedding: EmbeddingClient,
  index: IndexManager,
  options: SearchOptions & {
    vectorWeight?: number;
    bm25Weight?: number;
    rrfK?: number;
    timeDecay?: { enabled: boolean; halfLifeDays: number };
    maxSessionShare?: number;
    reranker?: Reranker | null;
  },
): Promise<SearchResult[]> {
  const normalizedQuery = query.trim();
  if (normalizedQuery.length === 0) {
    return [];
  }

  const maxResults = options.maxResults ?? DEFAULT_MAX_RESULTS;
  if (maxResults <= 0) {
    return [];
  }

  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  const vectorWeight = options.vectorWeight ?? DEFAULT_VECTOR_WEIGHT;
  const bm25Weight = options.bm25Weight ?? DEFAULT_BM25_WEIGHT;
  const rrfK = options.rrfK ?? DEFAULT_RRF_K;

  let queryVector: Float32Array | null = null;
  try {
    queryVector = await embedding.embedText(normalizedQuery, "RETRIEVAL_QUERY");
  } catch {
    queryVector = null;
  }

  const bm25Promise = Promise.resolve()
    .then(() => index.searchBM25(normalizedQuery, CANDIDATE_TOP_K))
    .catch(() => [] as ScoredChunk[]);
  const vectorPromise = queryVector
    ? Promise.resolve()
        .then(() => index.searchVector(queryVector, CANDIDATE_TOP_K))
        .catch(() => [] as ScoredChunk[])
    : Promise.resolve([] as ScoredChunk[]);

  const [bm25Results, vectorResults] = await Promise.all([bm25Promise, vectorPromise]);

  if (bm25Results.length === 0 && vectorResults.length === 0) {
    return [];
  }

  const bm25Scores = computeRRFScores(bm25Results, bm25Weight, rrfK);
  const vectorScores = computeRRFScores(vectorResults, vectorWeight, rrfK);

  const merged = new Map<string, MergedRankedChunk>();
  const mergeScores = (
    scores: Map<string, RankedChunk>,
    flags: Pick<MergedRankedChunk, "inBM25" | "inVector">,
  ): void => {
    for (const [key, value] of scores) {
      const existing = merged.get(key);
      if (existing) {
        existing.score += value.score;
        existing.inBM25 ||= flags.inBM25;
        existing.inVector ||= flags.inVector;
      } else {
        merged.set(key, {
          chunk: value.chunk,
          score: value.score,
          inBM25: flags.inBM25,
          inVector: flags.inVector,
        });
      }
    }
  };

  mergeScores(bm25Scores, { inBM25: true, inVector: false });
  mergeScores(vectorScores, { inBM25: false, inVector: true });

  const fused: RankedChunk[] = [];
  for (const entry of merged.values()) {
    if (entry.inBM25 && entry.inVector) {
      entry.score += CO_OCCURRENCE_BONUS;
    }
    fused.push({ chunk: entry.chunk, score: entry.score });
  }

  fused.sort((a, b) => b.score - a.score);

  if (options.reranker && fused.length > 0) {
    const topK = Math.min(RERANK_TOP_K, fused.length);
    const topCandidates = fused.slice(0, topK);

    try {
      const reranked = await options.reranker.rerank(
        normalizedQuery,
        topCandidates.map((row) => ({
          text: row.chunk.text,
          score: row.score,
        })),
      );

      if (reranked.length > 0) {
        const candidatesByText = new Map<string, RankedChunk[]>();
        for (const row of topCandidates) {
          const list = candidatesByText.get(row.chunk.text);
          if (list) {
            list.push(row);
          } else {
            candidatesByText.set(row.chunk.text, [row]);
          }
        }

        const rerankedRows: RankedChunk[] = [];
        for (const row of reranked) {
          const matches = candidatesByText.get(row.text);
          const match = matches?.shift();
          if (!match) {
            continue;
          }
          match.score = row.rerankerScore;
          rerankedRows.push(match);
        }

        const consumed = new Set(rerankedRows);
        for (const row of topCandidates) {
          if (!consumed.has(row)) {
            rerankedRows.push(row);
          }
        }

        const rest = fused.slice(topK);
        fused.length = 0;
        fused.push(...rerankedRows, ...rest);
      }
    } catch {
      // Ignore reranker failures and keep RRF ordering.
    }
  }

  if (options.timeDecay?.enabled === true) {
    applyTimeDecay(fused, options.timeDecay.halfLifeDays);
    fused.sort((a, b) => b.score - a.score);
  }

  const balanced =
    options.maxSessionShare === undefined
      ? fused
      : applySourceBalancing(fused, options.maxSessionShare, maxResults);

  return balanced
    .filter((row) => row.score >= minScore)
    .slice(0, maxResults)
    .map(({ chunk, score }) => ({
      path: chunk.fileKey,
      startLine: chunk.startLine,
      endLine: chunk.endLine,
      score,
      snippet: chunk.text,
      source: chunk.source,
      citation: `${chunk.fileKey}#L${chunk.startLine}-L${chunk.endLine}`,
    }));
}
