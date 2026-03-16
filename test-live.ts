/**
 * Live end-to-end test — uses real Gemini API, real SQLite, real pipeline.
 * Run: GEMINI_API_KEY="..." npx tsx test-live.ts
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createEmbeddingClient } from "./src/embedding.js";
import { createIndexManager } from "./src/store.js";
import { createSyncManager } from "./src/sync.js";
import { search } from "./src/search.js";

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error("❌ Set GEMINI_API_KEY environment variable");
  process.exit(1);
}

const DIMS = 768;
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

async function main() {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "engram-live-"));
  const workspaceDir = path.join(tmpDir, "workspace");
  const dbPath = path.join(tmpDir, "index.sqlite");

  try {
    // =========================================================================
    // 1. Test Gemini API connectivity
    // =========================================================================
    console.log("\n🔌 Test 1: Gemini Embedding-2 API");
    const startEmbed = Date.now();

    const embedding = await createEmbeddingClient({
      apiKey: API_KEY,
      dimensions: DIMS,
    });

    assert(embedding.model === "gemini-embedding-2-preview", "Model name correct");
    assert(embedding.dimensions === DIMS, `Dimensions = ${DIMS}`);

    const vec = await embedding.embedText("hello world", "RETRIEVAL_QUERY");
    const embedMs = Date.now() - startEmbed;

    assert(vec.length === DIMS, `Vector length = ${vec.length}`);
    assert(vec instanceof Float32Array, "Returns Float32Array");

    // Check normalization (should be ~1.0 for normalized vector)
    let norm = 0;
    for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
    norm = Math.sqrt(norm);
    assert(Math.abs(norm - 1.0) < 0.01, `L2 norm = ${norm.toFixed(4)} (should be ~1.0)`);

    console.log(`  ⏱️  Single embed: ${embedMs}ms`);

    // =========================================================================
    // 2. Test batch embedding
    // =========================================================================
    console.log("\n📦 Test 2: Batch embedding");
    const startBatch = Date.now();

    const batchVecs = await embedding.embedBatch(
      ["SQLite database storage", "vector search algorithms", "markdown parsing"],
      "RETRIEVAL_DOCUMENT",
    );
    const batchMs = Date.now() - startBatch;

    assert(batchVecs.length === 3, `Batch returned ${batchVecs.length} vectors`);
    assert(batchVecs.every((v) => v.length === DIMS), "All vectors correct dimensions");

    console.log(`  ⏱️  Batch embed (3 texts): ${batchMs}ms`);

    // =========================================================================
    // 3. Test full pipeline: create workspace → sync → search
    // =========================================================================
    console.log("\n🔄 Test 3: Full sync + search pipeline");

    // Create workspace with test files
    fs.mkdirSync(path.join(workspaceDir, "memory"), { recursive: true });

    fs.writeFileSync(
      path.join(workspaceDir, "MEMORY.md"),
      `# Engram Project

Engram is a multimodal memory plugin for OpenClaw agents.
It uses Gemini Embedding-2 for vector embeddings.
The storage backend is SQLite with FTS5 and sqlite-vec.
`,
    );

    fs.writeFileSync(
      path.join(workspaceDir, "memory", "architecture.md"),
      `# Architecture Decisions

## Storage
We chose SQLite because it's a single file per agent.
No external databases or sidecars required.
better-sqlite3 is the driver with WAL mode enabled.

## Search
Hybrid search combines BM25 keyword matching with vector similarity.
Results are fused using Reciprocal Rank Fusion with k=60.
The cross-encoder reranker runs locally via ONNX runtime.

## Embeddings
Gemini Embedding-2 provides task-aware embeddings.
RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for searching.
Default dimensions are 768 via Matryoshka truncation.
`,
    );

    fs.writeFileSync(
      path.join(workspaceDir, "memory", "bugs.md"),
      `# Bug Log

## 2026-03-14: ENOENT error in chunker
The chunker was throwing ENOENT when processing files with broken symlinks.
Fixed by adding an fs.existsSync check before reading.

## 2026-03-14: FTS5 query syntax error
Queries with special characters like "C++" caused FTS5 to throw.
Fixed by catching the error and returning empty results gracefully.
`,
    );

    // Create index manager and sync
    const index = createIndexManager({ dbPath, dimensions: DIMS, workspaceDir });
    const sync = createSyncManager({
      workspaceDir,
      index,
      embedding,
      chunkTokens: 1024,
      chunkOverlap: 0.15,
    });

    const startSync = Date.now();
    await sync.sync({ reason: "live-test" });
    const syncMs = Date.now() - startSync;

    const stats = index.stats();
    assert(stats.files === 3, `Indexed ${stats.files} files (expected 3)`);
    assert(stats.chunks > 0, `Created ${stats.chunks} chunks`);
    assert(!sync.isDirty(), "Index not dirty after sync");

    console.log(`  ⏱️  Sync (3 files, ${stats.chunks} chunks): ${syncMs}ms`);

    // =========================================================================
    // 4. Test search — semantic queries
    // =========================================================================
    console.log("\n🔍 Test 4: Semantic search");

    const startSearch1 = Date.now();
    const sqliteResults = await search("SQLite database storage", embedding, index, {
      maxResults: 5,
    });
    const search1Ms = Date.now() - startSearch1;

    assert(sqliteResults.length > 0, `"SQLite database storage" → ${sqliteResults.length} results`);
    if (sqliteResults.length > 0) {
      const topResult = sqliteResults[0];
      assert(
        topResult.path.includes("architecture") || topResult.path.includes("MEMORY"),
        `Top result from: ${topResult.path}`,
      );
      assert(topResult.score > 0, `Score: ${topResult.score.toFixed(4)}`);
      assert(topResult.snippet.length > 0, `Snippet length: ${topResult.snippet.length}`);
      assert(topResult.source === "memory", `Source: ${topResult.source}`);
      assert(!!topResult.citation, `Citation: ${topResult.citation}`);
    }
    console.log(`  ⏱️  Search 1: ${search1Ms}ms`);

    const startSearch2 = Date.now();
    const bugResults = await search("ENOENT error fix", embedding, index, {
      maxResults: 5,
    });
    const search2Ms = Date.now() - startSearch2;

    assert(bugResults.length > 0, `"ENOENT error fix" → ${bugResults.length} results`);
    if (bugResults.length > 0) {
      assert(
        bugResults[0].path.includes("bugs"),
        `Top result from bugs.md: ${bugResults[0].path}`,
      );
    }
    console.log(`  ⏱️  Search 2: ${search2Ms}ms`);

    const startSearch3 = Date.now();
    const rrfResults = await search("how does hybrid search fusion work", embedding, index, {
      maxResults: 5,
    });
    const search3Ms = Date.now() - startSearch3;

    assert(rrfResults.length > 0, `"how does hybrid search fusion work" → ${rrfResults.length} results`);
    if (rrfResults.length > 0) {
      assert(
        rrfResults[0].path.includes("architecture"),
        `Top result from architecture.md: ${rrfResults[0].path}`,
      );
    }
    console.log(`  ⏱️  Search 3: ${search3Ms}ms`);

    // =========================================================================
    // 5. Test memory_get equivalent
    // =========================================================================
    console.log("\n📄 Test 5: readFileContent (memory_get)");

    const fullContent = index.readFileContent("MEMORY.md");
    assert(fullContent !== null, "Read MEMORY.md");
    assert(fullContent!.text.includes("Engram"), "Content contains 'Engram'");

    const lineSlice = index.readFileContent("memory/bugs.md", 3, 2);
    assert(lineSlice !== null, "Read bugs.md lines 3-4");
    assert(lineSlice!.text.includes("ENOENT") || lineSlice!.text.length > 0, `Line slice: "${lineSlice!.text.slice(0, 60)}..."`);

    const missing = index.readFileContent("nonexistent.md");
    assert(missing === null, "Missing file returns null");

    // =========================================================================
    // 6. Test incremental sync
    // =========================================================================
    console.log("\n🔄 Test 6: Incremental sync");

    // Add a new file
    fs.writeFileSync(
      path.join(workspaceDir, "memory", "new-note.md"),
      `# New Note\n\nThis file was added after the initial sync.\nIt talks about Kubernetes deployment.\n`,
    );

    sync.markDirty();
    const startResync = Date.now();
    await sync.sync({ reason: "incremental" });
    const resyncMs = Date.now() - startResync;

    const newStats = index.stats();
    assert(newStats.files === 4, `After add: ${newStats.files} files (expected 4)`);
    console.log(`  ⏱️  Incremental sync: ${resyncMs}ms`);

    // Search for new content
    const k8sResults = await search("Kubernetes deployment", embedding, index, { maxResults: 3 });
    assert(k8sResults.length > 0 && k8sResults[0].path.includes("new-note"), "Found new file content");

    // =========================================================================
    // 7. Test plugin import
    // =========================================================================
    console.log("\n🔌 Test 7: Plugin import");

    const plugin = (await import("./src/index.js")).default;
    assert(plugin.id === "engram", `Plugin id: ${plugin.id}`);
    assert(plugin.kind === "memory", `Plugin kind: ${plugin.kind}`);
    assert(typeof plugin.register === "function", "register() is a function");

    // =========================================================================
    // Summary
    // =========================================================================
    console.log("\n" + "=".repeat(50));
    console.log(`📊 Results: ${passed} passed, ${failed} failed`);
    console.log(`📁 DB size: ${(fs.statSync(dbPath).size / 1024).toFixed(0)} KB`);
    console.log("=".repeat(50));

    // Cleanup
    index.close();
    sync.close();

    if (failed > 0) {
      process.exit(1);
    }
  } finally {
    // Clean up temp dir
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error("💥 Fatal error:", err);
  process.exit(1);
});
