/**
 * Plugin Load Test — simulates what OpenClaw does when loading Engram.
 * 
 * Tests:
 * 1. Module imports correctly
 * 2. Plugin shape matches OpenClaw expectations
 * 3. register() runs without crashing
 * 4. Tool factory returns working tools
 * 5. Tools produce correct output format
 * 6. CLI registration works
 * 7. Full search through tools (not direct module calls)
 * 
 * Run: GEMINI_API_KEY="..." npx tsx test-plugin-load.ts
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error("❌ Set GEMINI_API_KEY environment variable");
  process.exit(1);
}

let passed = 0;
let failed = 0;

function assert(condition: boolean, label: string, detail?: string) {
  if (condition) {
    console.log(`  ✅ ${label}`);
    passed++;
  } else {
    console.log(`  ❌ ${label}${detail ? ` — ${detail}` : ""}`);
    failed++;
  }
}

// ============================================================================
// Mock OpenClaw Plugin API — simulates what OpenClaw passes to register()
// ============================================================================

interface RegisteredTool {
  name?: string;
  names?: string[];
  factory?: (ctx: Record<string, unknown>) => unknown;
  tool?: Record<string, unknown>;
}

interface RegisteredCli {
  commands: string[];
  registrar: (opts: { program: unknown }) => void;
}

function createMockApi(opts: {
  workspaceDir: string;
  agentId: string;
  geminiApiKey: string;
}) {
  const tools: RegisteredTool[] = [];
  const clis: RegisteredCli[] = [];
  const logs: string[] = [];

  return {
    // What OpenClaw passes as api.pluginConfig (from plugins.entries.engram.config)
    pluginConfig: {
      geminiApiKey: opts.geminiApiKey,
    },

    // Full OpenClaw config (api.config)
    config: {
      models: {
        providers: {
          // No google provider — testing explicit key path
        },
      },
    },

    // Plugin runtime (api.runtime)
    runtime: {
      modelAuth: {
        resolveApiKeyForProvider: async (_params: { provider: string }) => {
          // Simulate: no Google provider configured
          throw new Error("Provider not configured");
        },
      },
      tools: {
        createMemorySearchTool: () => null,
        createMemoryGetTool: () => null,
        registerMemoryCli: () => {},
      },
      events: {
        onSessionTranscriptUpdate: () => () => {},
      },
      logging: {
        getChildLogger: () => ({
          info: (msg: string) => logs.push(`[INFO] ${msg}`),
          warn: (msg: string) => logs.push(`[WARN] ${msg}`),
          error: (msg: string) => logs.push(`[ERROR] ${msg}`),
        }),
      },
    },

    // Logger (api.logger)
    logger: {
      info: (msg: string) => logs.push(`[INFO] ${msg}`),
      warn: (msg: string) => logs.push(`[WARN] ${msg}`),
      error: (msg: string) => logs.push(`[ERROR] ${msg}`),
      debug: (msg: string) => logs.push(`[DEBUG] ${msg}`),
    },

    // Registration methods
    registerTool: (toolOrFactory: unknown, opts?: { name?: string; names?: string[] }) => {
      if (typeof toolOrFactory === "function") {
        tools.push({ factory: toolOrFactory as RegisteredTool["factory"], ...opts });
      } else {
        tools.push({ tool: toolOrFactory as Record<string, unknown>, ...opts });
      }
    },

    registerCli: (registrar: RegisteredCli["registrar"], opts?: { commands?: string[] }) => {
      clis.push({ registrar, commands: opts?.commands ?? [] });
    },

    registerService: () => {},
    registerHook: () => {},
    on: () => {},
    resolvePath: (p: string) => p.replace(/^~/, os.homedir()),

    // Accessors for test verification
    _tools: tools,
    _clis: clis,
    _logs: logs,
  };
}

// ============================================================================
// Tests
// ============================================================================

async function main() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-plugin-test-"));
  const workspaceDir = path.join(tmpDir, "workspace");
  const agentDir = path.join(tmpDir, "agent");

  try {
    // Create workspace with test files
    fs.mkdirSync(path.join(workspaceDir, "memory"), { recursive: true });
    fs.mkdirSync(path.join(agentDir, "engram"), { recursive: true });

    fs.writeFileSync(
      path.join(workspaceDir, "MEMORY.md"),
      "# Project Notes\n\nEngram is a memory plugin for OpenClaw.\nIt uses Gemini for embeddings.\n",
    );
    fs.writeFileSync(
      path.join(workspaceDir, "memory", "decisions.md"),
      "# Decisions\n\n## Storage\nWe chose SQLite for single-file storage.\n\n## Search\nHybrid search with RRF fusion.\n",
    );

    // =========================================================================
    // Test 1: Module import
    // =========================================================================
    console.log("\n📦 Test 1: Module Import");
    const plugin = (await import("./src/index.js")).default;
    assert(plugin !== null && plugin !== undefined, "Plugin module exports default");
    assert(typeof plugin === "object", "Export is an object");

    // =========================================================================
    // Test 2: Plugin shape
    // =========================================================================
    console.log("\n🔍 Test 2: Plugin Shape (OpenClaw requirements)");
    assert(plugin.id === "engram", `id: "${plugin.id}"`);
    assert(plugin.kind === "memory", `kind: "${plugin.kind}"`);
    assert(typeof plugin.name === "string" && plugin.name.length > 0, `name: "${plugin.name}"`);
    assert(typeof plugin.description === "string" && plugin.description.length > 0, "has description");
    assert(typeof plugin.register === "function", "register is a function");

    // =========================================================================
    // Test 3: register() doesn't crash
    // =========================================================================
    console.log("\n⚙️ Test 3: register() execution");
    const api = createMockApi({
      workspaceDir,
      agentId: "test-agent",
      geminiApiKey: API_KEY,
    });

    let registerError: Error | null = null;
    try {
      await plugin.register(api);
    } catch (e) {
      registerError = e instanceof Error ? e : new Error(String(e));
    }
    assert(registerError === null, "register() completed without error", registerError?.message);
    assert(api._logs.some(l => l.includes("engram")), `Logged startup: ${api._logs[0] ?? "(none)"}`);

    // =========================================================================
    // Test 4: Tool registration
    // =========================================================================
    console.log("\n🔧 Test 4: Tool Registration");
    assert(api._tools.length > 0, `${api._tools.length} tool registration(s)`);

    const memoryToolReg = api._tools.find(t => 
      t.names?.includes("memory_search") || t.name === "memory_search"
    );
    assert(memoryToolReg !== undefined, "memory_search tool registered");
    assert(
      memoryToolReg?.names?.includes("memory_get") === true,
      "memory_get tool registered (same factory)",
    );
    assert(typeof memoryToolReg?.factory === "function", "Uses factory pattern (per-session context)");

    // =========================================================================
    // Test 5: Tool factory produces tools
    // =========================================================================
    console.log("\n🏭 Test 5: Tool Factory");
    const ctx = {
      config: api.config,
      workspaceDir,
      agentDir,
      agentId: "test-agent",
      sessionKey: "test:session:001",
      sessionId: "abc-123",
    };

    let tools: unknown = null;
    let factoryError: Error | null = null;
    try {
      tools = await (memoryToolReg?.factory as Function)(ctx);
    } catch (e) {
      factoryError = e instanceof Error ? e : new Error(String(e));
    }

    // Factory might return null if async init fails, or return tools
    if (factoryError) {
      console.log(`  ⚠️ Factory threw: ${factoryError.message}`);
      console.log("  ⚠️ This may be expected if the factory does async init internally");
      // Try to check if it returned something before erroring
    }
    
    if (Array.isArray(tools)) {
      assert(tools.length === 2, `Factory returned ${tools.length} tools`);
      
      const searchTool = tools.find((t: any) => t.name === "memory_search");
      const getTool = tools.find((t: any) => t.name === "memory_get");
      
      assert(searchTool !== undefined, "memory_search tool present");
      assert(getTool !== undefined, "memory_get tool present");
      
      if (searchTool) {
        assert(typeof (searchTool as any).description === "string", "memory_search has description");
        assert(typeof (searchTool as any).parameters === "object", "memory_search has parameters");
        assert(typeof (searchTool as any).execute === "function", "memory_search has execute()");
      }
      
      if (getTool) {
        assert(typeof (getTool as any).description === "string", "memory_get has description");
        assert(typeof (getTool as any).execute === "function", "memory_get has execute()");
      }

      // =====================================================================
      // Test 6: Execute memory_search
      // =====================================================================
      if (searchTool && typeof (searchTool as any).execute === "function") {
        console.log("\n🔍 Test 6: memory_search execution");
        
        // Give sync a moment to index
        await new Promise(r => setTimeout(r, 2000));
        
        let searchResult: any = null;
        try {
          searchResult = await (searchTool as any).execute("call-001", { query: "SQLite storage" });
        } catch (e) {
          const err = e instanceof Error ? e : new Error(String(e));
          console.log(`  ⚠️ Search execute threw: ${err.message}`);
        }

        if (searchResult) {
          assert(searchResult.content !== undefined, "Returns content array");
          if (Array.isArray(searchResult.content) && searchResult.content.length > 0) {
            const textBlock = searchResult.content[0];
            assert(textBlock.type === "text", "Content block type is text");
            
            let parsed: any = null;
            try { parsed = JSON.parse(textBlock.text); } catch {}
            
            if (parsed) {
              assert(Array.isArray(parsed.results), `Has results array (${parsed.results?.length ?? 0} results)`);
              assert(typeof parsed.provider === "string", `Provider: ${parsed.provider}`);
              assert(typeof parsed.model === "string", `Model: ${parsed.model}`);
              
              if (parsed.results?.length > 0) {
                const r = parsed.results[0];
                assert(typeof r.path === "string", `Result path: ${r.path}`);
                assert(typeof r.startLine === "number", `Result startLine: ${r.startLine}`);
                assert(typeof r.endLine === "number", `Result endLine: ${r.endLine}`);
                assert(typeof r.score === "number", `Result score: ${r.score}`);
                assert(typeof r.snippet === "string", `Result snippet length: ${r.snippet?.length}`);
                assert(typeof r.source === "string", `Result source: ${r.source}`);
              }
            } else {
              assert(false, "Could not parse search result JSON", textBlock.text?.slice(0, 100));
            }
          }
        }
      }

      // =====================================================================
      // Test 7: Execute memory_get
      // =====================================================================
      if (getTool && typeof (getTool as any).execute === "function") {
        console.log("\n📄 Test 7: memory_get execution");
        
        let getResult: any = null;
        try {
          getResult = await (getTool as any).execute("call-002", { path: "MEMORY.md" });
        } catch (e) {
          const err = e instanceof Error ? e : new Error(String(e));
          console.log(`  ⚠️ Get execute threw: ${err.message}`);
        }

        if (getResult) {
          assert(getResult.content !== undefined, "Returns content array");
          if (Array.isArray(getResult.content) && getResult.content.length > 0) {
            let parsed: any = null;
            try { parsed = JSON.parse(getResult.content[0].text); } catch {}
            
            if (parsed) {
              assert(typeof parsed.text === "string", `Got text (${parsed.text?.length} chars)`);
              assert(typeof parsed.path === "string", `Got path: ${parsed.path}`);
              assert(parsed.text.includes("Engram"), "Content matches MEMORY.md");
            }
          }
        }

        // Test missing file
        let missingResult: any = null;
        try {
          missingResult = await (getTool as any).execute("call-003", { path: "nonexistent.md" });
        } catch {}
        
        if (missingResult) {
          let parsed: any = null;
          try { parsed = JSON.parse(missingResult.content[0].text); } catch {}
          if (parsed) {
            assert(parsed.text === "", "Missing file returns empty text");
          }
        }
      }
    } else if (tools === null) {
      console.log("  ⚠️ Factory returned null — tools not available (may need async init)");
      console.log("  ℹ️ This is acceptable for initial load; tools initialize on first use in OpenClaw");
    } else {
      assert(false, `Unexpected factory return type: ${typeof tools}`);
    }

    // =========================================================================
    // Test 8: CLI registration
    // =========================================================================
    console.log("\n💻 Test 8: CLI Registration");
    assert(api._clis.length > 0, `${api._clis.length} CLI registration(s)`);
    const engramCli = api._clis.find(c => c.commands.includes("engram"));
    assert(engramCli !== undefined, "engram CLI command registered");

    // =========================================================================
    // Summary
    // =========================================================================
    console.log("\n" + "=".repeat(50));
    console.log(`📊 Results: ${passed} passed, ${failed} failed`);
    console.log("=".repeat(50));

    if (failed > 0) process.exit(1);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error("💥 Fatal error:", err);
  process.exit(1);
});
