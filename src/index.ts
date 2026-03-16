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

interface CliAgentConfig {
  id: string;
  workspace: string;
  agentDir: string;
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

export function resolveAgentsFromConfig(clawConfig: any, agentFilter?: string): CliAgentConfig[] {
  const agentList = Array.isArray(clawConfig?.agents?.list) ? clawConfig.agents.list : [];
  const defaultWorkspace =
    clawConfig?.agents?.defaults?.workspace ?? path.join(homedir(), ".openclaw", "workspace");

  const agents: CliAgentConfig[] = agentList
    .filter((agent: any) => typeof agent?.id === "string" && agent.id.trim().length > 0)
    .map((agent: any): CliAgentConfig => ({
      id: String(agent.id),
      workspace: agent.workspace ?? defaultWorkspace,
      agentDir: agent.agentDir ?? defaultAgentDir(agent.id),
    }));

  if (agentFilter) {
    const match = agents.find((agent) => agent.id === agentFilter);
    if (!match) {
      throw new Error(`Agent "${agentFilter}" not found in config`);
    }
    return [match];
  }

  return agents;
}

function parsePositiveInteger(value: unknown, fallback: number): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (Number.isFinite(parsed) && parsed > 0) {
    return parsed;
  }
  return fallback;
}

function parseNonNegativeNumber(value: unknown, fallback: number): number {
  const parsed = Number.parseFloat(String(value ?? ""));
  if (Number.isFinite(parsed) && parsed >= 0) {
    return parsed;
  }
  return fallback;
}

function truncateSnippet(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength)}...`;
}

async function closeCliManagers(
  logger: { warn?: (message: string) => void } | undefined,
  agentIds: Iterable<string>,
): Promise<void> {
  for (const agentId of agentIds) {
    const manager = managers.get(agentId);
    if (!manager) {
      continue;
    }
    try {
      await manager.close();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      logger?.warn?.(`engram: failed to close manager for "${agentId}": ${message}`);
    } finally {
      managers.delete(agentId);
    }
  }
}

function formatSources(status: { sources: Array<{ source: string; files: number; chunks: number }> }): string {
  if (!Array.isArray(status.sources) || status.sources.length === 0) {
    return "none";
  }
  return status.sources.map((source) => source.source).join(", ");
}

function createCliContext(agent: CliAgentConfig): OpenClawContext {
  return {
    agentId: agent.id,
    agentDir: agent.agentDir,
    workspaceDir: agent.workspace,
  };
}

export function registerEngramCli(
  program: any,
  api: OpenClawApi,
  config: ResolvedConfig,
  clawConfig: any,
): void {
  const memory = program.command("engram").description("Engram memory plugin — status, search, and reindex");

  memory
    .command("status")
    .description("Show memory search index status")
    .option("--agent <id>", "Agent id")
    .option("--deep", "Probe embedding and vector availability", false)
    .option("--json", "Print JSON", false)
    .option("--verbose", "Verbose logging", false)
    .action(
      async (opts: { agent?: string; deep?: boolean; json?: boolean; verbose?: boolean }) => {
        let agents: CliAgentConfig[];
        try {
          agents = resolveAgentsFromConfig(clawConfig, opts.agent);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          console.error(message);
          process.exitCode = 1;
          return;
        }

        if (agents.length === 0) {
          console.error("No agents found in OpenClaw config (agents.list).");
          process.exitCode = 1;
          return;
        }

        const rows: Array<Record<string, unknown>> = [];
        const usedManagers = new Set<string>();

        try {
          for (const agent of agents) {
            if (opts.verbose) {
              api.logger?.info?.(`engram: status for agent "${agent.id}"`);
            }

            try {
              const manager = await getOrCreateEngramManager(api, config, createCliContext(agent));
              usedManagers.add(agent.id);

              const status = manager.status();
              let embeddingProbe: { ok: boolean; error?: string } | undefined;
              let vectorProbe: boolean | undefined;

              if (opts.deep) {
                embeddingProbe = await manager.probeEmbeddingAvailability();
                vectorProbe = await manager.probeVectorAvailability();
              }

              const output = {
                agentId: agent.id,
                provider: status.provider,
                model: status.model,
                files: status.files,
                chunks: status.chunks,
                dirty: status.dirty,
                dbPath: status.dbPath,
                workspaceDir: agent.workspace,
                sources: status.sources,
                vector: status.vector,
                fts: status.fts,
                embeddingProbe,
                vectorProbe,
              };

              rows.push(output);

              if (!opts.json) {
                console.log(`Memory Search (${agent.id})`);
                console.log(`Provider: ${status.provider} (Engram)`);
                console.log(`Model: ${status.model}`);
                console.log(`Sources: ${formatSources(status)}`);
                console.log(`Indexed: ${status.files} files · ${status.chunks} chunks`);
                console.log(`Dirty: ${status.dirty ? "yes" : "no"}`);
                console.log(`Store: ${status.dbPath}`);
                console.log(`Workspace: ${agent.workspace}`);
                console.log("By source:");
                for (const source of status.sources) {
                  console.log(`  ${source.source} · ${source.files} files · ${source.chunks} chunks`);
                }
                console.log(`Vector: ${status.vector.enabled ? "enabled" : "disabled"} (dims=${status.vector.dims})`);
                console.log(`FTS: ${status.fts.enabled ? "enabled" : "disabled"}`);
                if (opts.deep) {
                  const embeddingLine = embeddingProbe?.ok
                    ? "ok"
                    : `failed (${embeddingProbe?.error ?? "unknown error"})`;
                  console.log(`Embedding: ${embeddingLine}`);
                  console.log(`Vector probe: ${vectorProbe ? "ok" : "unavailable"}`);
                }
                console.log("");
              }
            } catch (error) {
              const message = error instanceof Error ? error.message : String(error);
              rows.push({ agentId: agent.id, error: message });
              if (opts.json) {
                continue;
              }
              console.error(`Memory status failed (${agent.id}): ${message}`);
              console.log("");
              process.exitCode = 1;
            }
          }

          if (opts.json) {
            console.log(JSON.stringify(rows, null, 2));
          }
        } finally {
          await closeCliManagers(api.logger, usedManagers);
        }
      },
    );

  memory
    .command("index")
    .description("Reindex memory files")
    .option("--agent <id>", "Agent id")
    .option("--force", "Force full reindex", false)
    .option("--verbose", "Verbose logging", false)
    .action(async (opts: { agent?: string; force?: boolean; verbose?: boolean }) => {
      let agents: CliAgentConfig[];
      try {
        agents = resolveAgentsFromConfig(clawConfig, opts.agent);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(message);
        process.exitCode = 1;
        return;
      }

      if (agents.length === 0) {
        console.error("No agents found in OpenClaw config (agents.list).");
        process.exitCode = 1;
        return;
      }

      let completed = 0;
      const usedManagers = new Set<string>();

      try {
        for (const agent of agents) {
          if (opts.verbose) {
            api.logger?.info?.(`engram: indexing agent "${agent.id}"`);
          }

          process.stdout.write(`Indexing ${agent.id}... `);
          try {
            const manager = await getOrCreateEngramManager(api, config, createCliContext(agent));
            usedManagers.add(agent.id);
            await manager.sync({ force: Boolean(opts.force) });
            const status = manager.status();
            console.log(`done (${status.files} files, ${status.chunks} chunks)`);
            completed += 1;
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            console.log(`failed (${message})`);
            process.exitCode = 1;
          }
        }
      } finally {
        await closeCliManagers(api.logger, usedManagers);
      }

      console.log(`Indexed ${completed} agents`);
    });

  memory
    .command("search")
    .description("Search memory files")
    .argument("[query]", "Search query")
    .option("--agent <id>", "Agent id")
    .option("--query <text>", "Search query (alternative to positional argument)")
    .option("--max-results <n>", "Max results", "10")
    .option("--min-score <n>", "Minimum score", "0")
    .option("--json", "Print JSON", false)
    .action(
      async (
        queryArg: string | undefined,
        opts: {
          agent?: string;
          query?: string;
          maxResults?: string;
          minScore?: string;
          json?: boolean;
        },
      ) => {
        const query = (opts.query ?? queryArg ?? "").trim();
        if (!query) {
          console.error("Missing search query. Provide [query] or --query <text>.");
          process.exitCode = 1;
          return;
        }

        let agent: CliAgentConfig | undefined;
        try {
          if (opts.agent) {
            agent = resolveAgentsFromConfig(clawConfig, opts.agent)[0];
          } else {
            const agents = resolveAgentsFromConfig(clawConfig);
            agent = agents[0];
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          console.error(message);
          process.exitCode = 1;
          return;
        }

        if (!agent) {
          console.error("No agents found in OpenClaw config (agents.list).");
          process.exitCode = 1;
          return;
        }

        const maxResults = parsePositiveInteger(opts.maxResults, 10);
        const minScore = parseNonNegativeNumber(opts.minScore, 0);

        const usedManagers = new Set<string>();
        try {
          const manager = await getOrCreateEngramManager(api, config, createCliContext(agent));
          usedManagers.add(agent.id);

          const results = await manager.search(query, {
            maxResults,
            minScore,
          });

          if (opts.json) {
            console.log(JSON.stringify(results, null, 2));
            return;
          }

          if (results.length === 0) {
            console.log("No matches.");
            return;
          }

          for (const result of results) {
            const citation = `${result.path}#L${result.startLine}-L${result.endLine}`;
            console.log(`[${result.score.toFixed(2)}] ${citation}`);
            console.log(`  ${truncateSnippet(result.snippet, 200)}`);
            console.log("");
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          console.error(`Memory search failed: ${message}`);
          process.exitCode = 1;
        } finally {
          await closeCliManagers(api.logger, usedManagers);
        }
      },
    );
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
        ({ program, config: clawConfig }: any) => {
          registerEngramCli(program, api, config, clawConfig);
        },
        { commands: ["engram"] },
      );
    } else {
      logger?.warn?.("engram: registerCli is not available on plugin API");
    }
  },
};

export default engramPlugin;
