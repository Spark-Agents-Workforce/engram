export interface Reranker {
  /** Rerank candidates by (query, document) relevance. Returns reordered with updated scores. */
  rerank(
    query: string,
    candidates: Array<{ text: string; score: number }>,
  ): Promise<Array<{ text: string; score: number; rerankerScore: number }>>;

  /** Clean up model resources */
  close(): Promise<void>;
}

const MODEL_NAME = "Xenova/ms-marco-MiniLM-L-12-v2";
const MAX_INPUT_LENGTH = 512;
const MIN_QUERY_TERM_LEN = 3;

type RerankCandidate = { text: string; score: number };
type RerankedCandidate = { text: string; score: number; rerankerScore: number };

function toTerms(text: string): string[] {
  return text
    .toLowerCase()
    .split(/\W+/)
    .filter((term) => term.length >= MIN_QUERY_TERM_LEN);
}

function extractRerankerScores(
  logits: { data?: ArrayLike<number>; dims?: ArrayLike<number> } | undefined,
  candidateCount: number,
): number[] {
  if (candidateCount <= 0) {
    return [];
  }

  const rawData = logits?.data ? Array.from(logits.data) : [];
  if (rawData.length === 0) {
    return Array.from({ length: candidateCount }, () => Number.NEGATIVE_INFINITY);
  }

  const dims = logits?.dims ? Array.from(logits.dims, (value) => Number(value)) : [];
  const classesPerCandidate =
    dims.length >= 2 ? Math.max(1, dims[dims.length - 1] ?? 1) : Math.max(1, Math.floor(rawData.length / candidateCount));

  const scores: number[] = [];
  for (let i = 0; i < candidateCount; i += 1) {
    const offset = i * classesPerCandidate;
    if (classesPerCandidate === 1) {
      scores.push(rawData[offset] ?? rawData[i] ?? Number.NEGATIVE_INFINITY);
      continue;
    }

    // For two-class heads, use the positive class logit (last class).
    scores.push(rawData[offset + classesPerCandidate - 1] ?? Number.NEGATIVE_INFINITY);
  }

  return scores;
}

export function createLightweightReranker(): Reranker {
  return {
    async rerank(query: string, candidates: RerankCandidate[]): Promise<RerankedCandidate[]> {
      if (candidates.length === 0) {
        return [];
      }

      const queryTerms = new Set(toTerms(query));
      const reranked = candidates.map((candidate) => {
        const docTerms = new Set(toTerms(candidate.text));
        const overlapCount = Array.from(queryTerms).filter((term) => docTerms.has(term)).length;
        const coverage = queryTerms.size > 0 ? overlapCount / queryTerms.size : 0;
        const rerankerScore = 0.6 * candidate.score + 0.4 * coverage;

        return {
          text: candidate.text,
          score: candidate.score,
          rerankerScore,
        };
      });

      reranked.sort((a, b) => b.rerankerScore - a.rerankerScore);
      return reranked;
    },

    async close(): Promise<void> {},
  };
}

export async function createReranker(): Promise<Reranker | null> {
  try {
    // Dynamic import — @xenova/transformers bundles sharp which may not have
    // a working native binary on all platforms. Loading lazily means the plugin
    // still works even if the binary is missing (falls back to lightweight reranker).
    const { AutoModelForSequenceClassification, AutoTokenizer, env } = await import("@xenova/transformers");

    if (process.env.NODE_ENV === "production") {
      env.allowRemoteModels = false;
    }
    const localFilesOnly = env.allowRemoteModels === false;

    const tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME, {
      local_files_only: localFilesOnly,
    });
    const model = await AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, {
      local_files_only: localFilesOnly,
    });

    return {
      async rerank(query: string, candidates: RerankCandidate[]): Promise<RerankedCandidate[]> {
        if (candidates.length === 0) {
          return [];
        }

        const queries = candidates.map(() => query);
        const documents = candidates.map((candidate) => candidate.text);
        const inputs = tokenizer(queries, {
          text_pair: documents,
          padding: true,
          truncation: true,
          max_length: MAX_INPUT_LENGTH,
        });
        const outputs = await model(inputs);

        const scores = extractRerankerScores(outputs?.logits, candidates.length);
        const reranked = candidates.map((candidate, i) => ({
          text: candidate.text,
          score: candidate.score,
          rerankerScore: scores[i] ?? Number.NEGATIVE_INFINITY,
        }));

        reranked.sort((a, b) => b.rerankerScore - a.rerankerScore);
        return reranked;
      },

      async close(): Promise<void> {
        if (typeof model.dispose === "function") {
          await model.dispose();
        }
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`Engram: Cross-encoder reranker unavailable: ${message}`);
    return null;
  }
}
