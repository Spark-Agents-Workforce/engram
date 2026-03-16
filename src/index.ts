import { homedir } from "node:os";
import path from "node:path";
import { resolveConfig, type ResolvedConfig } from "./config.js";
import { createEmbeddingClient } from "./embedding.js";
import { createEngramManager, type EngramManager } from "./manager.js";
import { createReranker } from "./reranker.js";
import { createIndexManager } from "./store.js";
import { createSyncManager } from "./sync.js";

type OpenClawApi = any;
type OpenClawContext = any;

const managers = new Map<string, EngramManager>();

function defaultAgentDir(agentId: string): string {
  return path.join(homedir(), ".openclaw", "agents", agentId);
}

function defaultCliContext(): OpenClawContext {
  return {
    agentId: "default",
    agentDir: defaultAgentDir("default"),
    workspaceDir: process.cwd(),
  };
}

async function resolveApiKey(api: OpenClawApi, config: ResolvedConfig): Promise<string> {
  if (config.geminiApiKey) {
    const envMatch = config.geminiApiKey.match(/^\$\{(\w+)\}$/);
    if (envMatch) {
      const envValue = process.env[envMatch[1]];
      if (envValue) {
        return envValue;
      }
    } else {
      return config.geminiApiKey;
    }
  }

  try {
    const auth = await api.runtime.modelAuth.resolveApiKeyForProvider({
      provider: "google",
      cfg: api.config,
    });
    if (auth?.apiKey) {
      return auth.apiKey;
    }
  } catch {
    // Provider auth may be unavailable depending on host runtime.
  }

  if (process.env.GEMINI_API_KEY) {
    return process.env.GEMINI_API_KEY;
  }
  if (process.env.GOOGLE_API_KEY) {
    return process.env.GOOGLE_API_KEY;
  }

  throw new Error(
    "Engram: No Gemini API key found. Either configure Google as a provider (openclaw onboard), " +
      "set GEMINI_API_KEY environment variable, or add geminiApiKey to plugin config.",
  );
}

async function getOrCreateEngramManager(
  api: OpenClawApi,
  config: ResolvedConfig,
  ctx: OpenClawContext,
): Promise<EngramManager> {
  const agentId = String(ctx?.agentId ?? "default");
  const existing = managers.get(agentId);
  if (existing) {
    return existing;
  }

  const apiKey = await resolveApiKey(api, config);
  const baseUrl = api?.config?.models?.providers?.google?.baseUrl as string | undefined;
  const workspaceDir = (ctx?.workspaceDir as string | undefined) ?? process.cwd();

  const embedding = await createEmbeddingClient({
    apiKey,
    dimensions: config.dimensions,
    baseUrl,
  });

  const dbPath = path.join(
    (ctx?.agentDir as string | undefined) ?? defaultAgentDir(agentId),
    "engram",
    "index.sqlite",
  );
  const sessionsDir = path.join(
    (ctx?.agentDir as string | undefined) ?? defaultAgentDir(agentId),
    "sessions",
  );

  const index = createIndexManager({
    dbPath,
    dimensions: config.dimensions,
    workspaceDir,
  });

  const syncManager = createSyncManager({
    workspaceDir,
    index,
    embedding,
    chunkTokens: config.chunkTokens,
    chunkOverlap: config.chunkOverlap,
    sessionsDir,
    multimodal: config.multimodal.enabled
      ? {
          enabled: true,
          modalities: config.multimodal.modalities,
          maxFileBytes: config.multimodal.maxFileBytes,
        }
      : undefined,
  });

  const reranker = config.reranking ? await createReranker() : null;
  if (reranker) {
    api.logger?.info?.("engram: cross-encoder reranker loaded");
  } else if (config.reranking) {
    api.logger?.warn?.("engram: reranker requested but unavailable, continuing without");
  }

  const manager = createEngramManager({
    index,
    embedding,
    syncManager,
    reranker,
    config,
    workspaceDir,
  });

  syncManager.startWatching({
    debounceMs: 1000,
    intervalMinutes: 5,
  });

  managers.set(agentId, manager);
  api.logger?.info?.(`engram: initialized manager for agent "${agentId}"`);
  return manager;
}

function stringifyToolOutput(value: unknown): {
  content: Array<{ type: "text"; text: string }>;
  details: unknown;
} {
  return {
    content: [{ type: "text", text: JSON.stringify(value, null, 2) }],
    details: value,
  };
}

function buildMemorySearchTool(
  api: OpenClawApi,
  getManager: () => Promise<EngramManager>,
  ctx: OpenClawContext,
) {
  return {
    name: "memory_search",
    label: "Memory Search",
    description:
      "Semantically search memory files and session transcripts. Returns top snippets with path and lines.",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query" },
        maxResults: { type: "number", description: "Max results (default: 10)" },
        minScore: { type: "number", description: "Min relevance score 0-1 (default: 0)" },
      },
      required: ["query"],
    },
    async execute(
      _toolCallId: string,
      params: { query: string; maxResults?: number; minScore?: number },
    ): Promise<{ content: Array<{ type: "text"; text: string }> }> {
      try {
        const manager = await getManager();
        manager.syncIfDirty();
        const results = await manager.search(params.query, {
          maxResults: params.maxResults,
          minScore: params.minScore,
          sessionKey: ctx?.sessionKey as string | undefined,
        });
        const status = manager.status();

        return stringifyToolOutput({
          results,
          provider: status.provider,
          model: status.model,
          citations: "auto",
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        api.logger?.error?.(`engram: memory_search failed: ${message}`);
        return stringifyToolOutput({
          results: [],
          disabled: true,
          unavailable: true,
          error: message,
          warning: `Engram search failed: ${message}`,
          action: "Check Gemini API key configuration",
        });
      }
    },
  };
}

function buildMemoryGetTool(api: OpenClawApi, getManager: () => Promise<EngramManager>) {
  return {
    name: "memory_get",
    label: "Memory Get",
    description: "Read a snippet from a memory file by path and optional line range.",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Relative file path" },
        from: { type: "number", description: "Start line (1-indexed)" },
        lines: { type: "number", description: "Number of lines to return" },
      },
      required: ["path"],
    },
    async execute(
      _toolCallId: string,
      params: { path: string; from?: number; lines?: number },
    ): Promise<{ content: Array<{ type: "text"; text: string }> }> {
      try {
        const manager = await getManager();
        const result = await manager.readFile({
          relPath: params.path,
          from: params.from,
          lines: params.lines,
        });

        return stringifyToolOutput({
          text: result.text,
          path: result.path,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        api.logger?.error?.(`engram: memory_get failed for "${params.path}": ${message}`);
        return stringifyToolOutput({
          path: params.path,
          text: "",
          disabled: true,
          unavailable: true,
          error: message,
        });
      }
    },
  };
}

const engramPlugin = {
  id: "engram",
  name: "Engram",
  description:
    "Multimodal memory powered by Gemini Embedding-2 with hybrid search and ONNX reranking",
  kind: "memory" as const,

  register(api: OpenClawApi) {
    const config = resolveConfig(api.pluginConfig);
    const logger = api.logger;

    logger?.info?.(
      `engram: registered (dims: ${config.dimensions}, chunks: ${config.chunkTokens})`,
    );

    if (typeof api.registerTool === "function") {
      api.registerTool(
        (ctx: OpenClawContext) => {
          const manager = getOrCreateEngramManager(api, config, ctx);
          return [
            buildMemorySearchTool(api, () => manager, ctx),
            buildMemoryGetTool(api, () => manager),
          ];
        },
        { names: ["memory_search", "memory_get"] },
      );
    } else {
      logger?.warn?.("engram: registerTool is not available on plugin API");
    }

    if (typeof api.registerCli === "function") {
      api.registerCli(
        ({ program }: any) => {
          const cmd = program.command("engram").description("Engram memory plugin");

          cmd
            .command("status")
            .description("Show Engram index status")
            .action(async () => {
              try {
                const manager = await getOrCreateEngramManager(api, config, defaultCliContext());
                const status = manager.status();
                const embeddingProbe = await manager.probeEmbeddingAvailability();
                const vectorProbe = await manager.probeVectorAvailability();

                console.log(`Backend: ${status.backend}`);
                console.log(`Provider: ${status.provider}`);
                console.log(`Model: ${status.model}`);
                console.log(`Workspace: ${status.workspaceDir}`);
                console.log(`DB: ${status.dbPath}`);
                console.log(`Files: ${status.files}`);
                console.log(`Chunks: ${status.chunks}`);
                console.log(`Dirty: ${status.dirty ? "yes" : "no"}`);
                console.log(`Vector: ${status.vector.enabled ? "enabled" : "disabled"} (dims=${status.vector.dims})`);
                console.log(`FTS: ${status.fts.enabled ? "enabled" : "disabled"} (available=${status.fts.available})`);
                console.log(
                  `Scoring: rrfK=${status.custom.rrfK}, vectorWeight=${status.custom.vectorWeight}, bm25Weight=${status.custom.bm25Weight}`,
                );
                console.log(
                  `Chunking: tokens=${status.custom.chunkTokens}, overlap=${status.custom.chunkOverlap}`,
                );
                console.log("Sources:");
                for (const source of status.sources) {
                  console.log(`- ${source.source}: files=${source.files}, chunks=${source.chunks}`);
                }
                console.log(
                  `Embedding probe: ${embeddingProbe.ok ? "ok" : `failed (${embeddingProbe.error ?? "unknown error"})`}`,
                );
                console.log(`Vector probe: ${vectorProbe ? "ok" : "unavailable"}`);
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                console.error(`Engram status failed: ${message}`);
                process.exitCode = 1;
              }
            });

          cmd
            .command("search")
            .description("Test search")
            .argument("<query>")
            .option("--limit <n>", "Max results", "5")
            .action(async (query: string, opts: { limit?: string }) => {
              try {
                const manager = await getOrCreateEngramManager(api, config, defaultCliContext());
                const parsed = Number.parseInt(opts.limit ?? "5", 10);
                const limit = Number.isFinite(parsed) && parsed > 0 ? parsed : 5;
                const results = await manager.search(query, { maxResults: limit });

                if (results.length === 0) {
                  console.log("No results.");
                  return;
                }

                for (let i = 0; i < results.length; i += 1) {
                  const row = results[i];
                  console.log(
                    `${i + 1}. ${row.path}:${row.startLine}-${row.endLine} score=${row.score.toFixed(4)} source=${row.source}`,
                  );
                  console.log(`   ${row.snippet}`);
                }
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                console.error(`Engram search failed: ${message}`);
                process.exitCode = 1;
              }
            });
        },
        { commands: ["engram"] },
      );
    } else {
      logger?.warn?.("engram: registerCli is not available on plugin API");
    }
  },
};

export default engramPlugin;
