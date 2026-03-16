<p align="center">
  <h1 align="center">💎 Engram</h1>
</p>

<p align="center">
  <strong>Memory search for OpenClaw that actually finds what you meant.</strong><br>
  <sub>Powered by Gemini Embedding-2 — text, images, and audio in one search space.<br>One config line. Zero migration. Switch back in 10 seconds.</sub>
</p>

<p align="center">
  <a href="https://github.com/Spark-Agents-Workforce/engram/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/OpenClaw-Plugin-FF6B35" alt="OpenClaw Plugin">
  <img src="https://img.shields.io/badge/Gemini-Embedding--2-4285F4?logo=google&logoColor=white" alt="Gemini Embedding-2">
  <img src="https://img.shields.io/badge/tests-98%20passing-brightgreen" alt="Tests passing">
</p>

<p align="center">
  <a href="#why">Why</a> •
  <a href="#install">Install</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#comparison">Comparison</a> •
  <a href="#faq">FAQ</a>
</p>

---

<a id="why"></a>

## Why Engram Exists

OpenClaw's builtin memory search uses a simple weighted average of keyword and vector scores. It works for easy queries. But it falls apart when:

- **Your wording changes.** You wrote "database error" — you search "that SQLite bug" — no match.
- **Old and new get mixed together.** You ask "what did we decide?" and get a brainstorm from two months ago instead of last week's decision.
- **You saved a screenshot, not text.** The builtin can't search images at all.

Engram replaces the retrieval pipeline with one built on **Google's Gemini Embedding-2** — the first embedding model that natively understands text, images, and audio in a single vector space. Not stitched together with separate models. One model, one space, one search.

```text
You: "Find that error we fixed last Tuesday"
Builtin: 🤷 random notes about errors from 3 months ago
Engram: ✅ memory/2026-03-12.md — "Fixed ENOENT crash in the chunker module"

You: "that architecture diagram from the planning session"
Builtin: ❌ can't search images
Engram: ✅ memory/architecture-sketch.png — matched by visual content

You: "what did the client say about pricing"
Builtin: ❌ buries last week's notes under older matches
Engram: ✅ memory/2026-03-10.md — recent results ranked first
```

### What makes it actually better

🔍 **Smarter retrieval**

Keyword search and semantic search run in parallel, then get fused with Reciprocal Rank Fusion — a ranking method that's more robust than a simple weighted average. The top candidates then pass through a cross-encoder reranker (running locally, no API call) that reads each query-document pair together and re-scores for precision.

🕐 **Time-aware**

Recently-indexed notes score higher than old ones. Exponential decay with a configurable half-life (30 days default). Files you actively edit stay fresh. Old notes you never touch gradually fade. Your agent gets what's current, not what's ancient.

🖼️ **Multimodal**

Gemini Embedding-2 puts text, images, and audio in the same vector space. A text query can find a screenshot. A description can surface a voice memo. No OCR, no transcription — the model understands the content natively. Supported: `.jpg`, `.png`, `.webp`, `.gif`, `.mp3`, `.wav`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.flac`. Multimodal is on by default — images and audio in your workspace get indexed automatically alongside your markdown.

📦 **Self-contained**

One SQLite file per agent. No Python. No sidecar process. No external database. No external API for reranking. Install it, configure one line, restart.

---

<a id="install"></a>

## Install

```bash
cd ~/.openclaw/extensions && npm install @sparkagents/engram
```

```yaml
plugins:
  slots:
    memory: "engram"
```

```bash
openclaw gateway restart
```

That's it. If you already use Gemini in OpenClaw, the API key is auto-detected. If not, get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (60 seconds) and set `GEMINI_API_KEY` in your environment.

> **Switching from QMD?** Also set `memory.backend: "builtin"` in your config (QMD overrides the plugin slot). Your QMD index stays on disk untouched — switch back anytime.

<details>
<summary>Other API key options</summary>

- Set `GOOGLE_API_KEY` in your environment
- Add `geminiApiKey: "${GEMINI_API_KEY}"` to the plugin config under `plugins.entries.engram.config`
- Run `openclaw onboard` and add Google as a provider
- If you use a restrictive `plugins.allow` list, add `"engram"` to it

</details>

### Switching back

Your files don't move. Nothing gets deleted. QMD's index stays untouched. Both indexes coexist.

```yaml
# Using Engram
plugins:
  slots:
    memory: "engram"

# Switch back — remove the lines above and restart
# Builtin or QMD picks up exactly where it left off
```

---

<a id="how-it-works"></a>

## How It Works

### Search pipeline

```text
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Query                                                       │
│    │                                                         │
│    ▼                                                         │
│  Embed query (Gemini Embedding-2, RETRIEVAL_QUERY)           │
│    │                                                         │
│    ├───────────────────┬──────────────────┐                  │
│    ▼                   ▼                  │                  │
│  BM25 keyword        Vector KNN           │                  │
│  search (FTS5)       search (sqlite-vec)  │                  │
│    │                   │                  │                  │
│    └─────────┬─────────┘                  │                  │
│              ▼                            │                  │
│  Reciprocal Rank Fusion (k=60)            │                  │
│  weight: 0.7 vector / 0.3 keyword        │                  │
│              │                            │                  │
│              ▼                            │                  │
│  Cross-encoder reranking (top 20)         │                  │
│  ms-marco-MiniLM-L-12-v2 — local ONNX    │                  │
│              │                            │                  │
│              ▼                            │                  │
│  Time decay + source balancing            │                  │
│              │                            │                  │
│              ▼                            │                  │
│  Results                                  │                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Indexing

1. Discovers `MEMORY.md`, `memory/*.md`, session transcripts (JSONL), and optionally image/audio files
2. Chunks markdown at 1024 tokens with 15% overlap — respects headings, code blocks, frontmatter
3. Embeds with Gemini Embedding-2 using `RETRIEVAL_DOCUMENT` task type (768 dimensions default, L2-normalized)
4. Stores in one SQLite file: FTS5 for keywords, sqlite-vec for vectors, metadata for freshness tracking
5. Incremental: content-hash tracking means only changed files get re-embedded
6. Live: chokidar watches workspace files and session transcript directory for real-time updates

### Performance

Measured on Apple Silicon, real Gemini API calls:

| Operation | Time |
|---|---|
| Search (full pipeline) | ~350ms |
| Index one file | ~400ms |
| First sync (small workspace) | ~1-2s |
| Local pipeline (FTS5 + vector + RRF + rerank) | <10ms |

Search latency is dominated by the Gemini API round-trip. Everything after the embedding call runs locally in under 10ms.

### Storage

One SQLite file per agent at `~/.openclaw/agents/{id}/engram/index.sqlite`.

| Workspace size | Chunks | Index |
|---|---|---|
| 100 files | ~2K | ~10 MB |
| 500 files | ~10K | ~50 MB |
| 2000 files | ~50K | ~250 MB |

---

<a id="comparison"></a>

## Comparison

| | Builtin | QMD | LanceDB Pro | **Engram** |
|---|---|---|---|---|
| **Multimodal** (image + audio search) | ❌ | ❌ | ❌ | **✅** |
| **Hybrid search** (keywords + vectors) | Weak | ✅ | ✅ | ✅ |
| **Reranking** | ❌ | ✅ local GGUF | ✅ external API | ✅ local ONNX |
| **Task-aware embeddings** | ❌ | ❌ | ❌ | **✅** |
| **Reciprocal Rank Fusion** | ❌ | ✅ | ❌ | ✅ |
| **Time decay** | ❌ | ❌ | ✅ | ✅ |
| **Session transcript search** | ✅ | ✅ | ❌ | ✅ |
| **Single file storage** | ✅ | ✅ | ❌ | ✅ |
| **No sidecar / external binary** | ✅ | ❌ | ✅ | ✅ |
| **No external API for reranking** | N/A | ✅ | ❌ | ✅ |
| **Zero-config startup** | ✅ | ⚠️ | ❌ | ✅ |

Engram is the only OpenClaw memory plugin that combines multimodal search, hybrid retrieval with RRF, local cross-encoder reranking, and single-file storage.

---

### Configuration

Everything below is optional. Defaults are tuned and work out of the box.

```yaml
plugins:
  slots:
    memory: "engram"
  entries:
    engram:
      config:
        # geminiApiKey: "${GEMINI_API_KEY}"     # only if Gemini isn't already configured
        # dimensions: 768                       # 768 (default) | 1536 | 3072
        # chunkTokens: 1024                     # 512-8192 (default: 1024)
        # chunkOverlap: 0.15                    # 0-0.5 (default: 0.15)
        # reranking: true                       # cross-encoder reranking (default: on)
        # timeDecay:
        #   enabled: true                       # recency boost (default: on)
        #   halfLifeDays: 30                    # score halves every 30 days
        # maxSessionShare: 0.4                  # cap session results at 40%
        # multimodal:
        #   enabled: true                       # index images and audio (default: on)
        #   modalities: ["image", "audio"]
```

### CLI

```bash
openclaw engram status                          # index stats for all agents
openclaw engram status --agent main             # single agent
openclaw engram status --agent main --deep      # probe embedding + vector availability
openclaw engram index --force                   # reindex all agents
openclaw engram index --agent main              # reindex single agent
openclaw engram search "query"                  # search (default agent)
openclaw engram search "query" --agent main     # search specific agent
openclaw engram search "query" --json           # machine-readable output
```

---

<a id="faq"></a>

## FAQ

<details>
<summary><strong>What happens to my existing memory files?</strong></summary>

Nothing. Your markdown files (`MEMORY.md`, `memory/*.md`) stay exactly where they are. Engram reads them — it doesn't move, copy, or modify them. Same files, better search.

</details>

<details>
<summary><strong>What about my QMD index?</strong></summary>

QMD's index is untouched. Engram builds a separate index in its own SQLite file. Both coexist on disk. Switching between them is one config line plus a restart.

</details>

<details>
<summary><strong>How long does the first sync take?</strong></summary>

For a typical workspace (20-50 markdown files), about 10-30 seconds. Engram chunks every file and sends the text to the Gemini API for embedding. After that, only changed files get re-embedded.

</details>

<details>
<summary><strong>Does this cost money?</strong></summary>

Gemini's embedding API has a free tier (~1,500 requests/day). Most personal setups will never hit the limit. If you do, text embeddings cost $0.20 per million tokens — a typical workspace costs less than a penny to index.

</details>

<details>
<summary><strong>What if I don't have a Gemini API key?</strong></summary>

Get a free one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Takes about 60 seconds.

</details>

<details>
<summary><strong>Is it slower than QMD?</strong></summary>

Yes. QMD runs 100% locally (~120ms). Engram makes one API call per search (~350ms). The tradeoff: Gemini Embedding-2 is a significantly stronger model than QMD's 300M-parameter local embeddings, and Engram supports multimodal. If latency matters more than accuracy, stick with QMD.

</details>

<details>
<summary><strong>Will this break my agents?</strong></summary>

No. Engram registers the same `memory_search` and `memory_get` tools your agents already use. From an agent's perspective, nothing changes except better results. If anything goes wrong, `memory_search` returns empty results gracefully instead of crashing.

</details>

<details>
<summary><strong>Can I use it for just one agent?</strong></summary>

Not currently. The memory plugin slot is global — applies to all agents on the gateway. That's an OpenClaw limitation, not Engram's. Per-agent overrides may come in a future OpenClaw version.

</details>

<details>
<summary><strong>What data gets sent to Google?</strong></summary>

Text content of your memory files and session transcripts is sent to Google's Gemini embedding API. If multimodal is enabled, image and audio bytes are sent too. Google's paid tier does not use your data for model training. Embeddings are one-way — original content can't be reconstructed from vectors.

</details>

<details>
<summary><strong>What about PDF search?</strong></summary>

Gemini Embedding-2 supports PDF embedding (up to 6 pages per request). Engram doesn't index PDFs yet but the model path is there — it's on the roadmap.

</details>

---

<p align="center">
  <strong>Ready?</strong> <code>npm install</code> → one config line → <code>openclaw gateway restart</code><br>
  <a href="#install">↑ Install now</a>
</p>

<p align="center">
  <sub>Built by <a href="https://sparkagents.com">Spark Agents</a> for the <a href="https://github.com/openclaw/openclaw">OpenClaw</a> community.</sub>
</p>
