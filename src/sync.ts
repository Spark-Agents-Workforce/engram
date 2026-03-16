import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { watch, type FSWatcher } from "chokidar";
import { chunkMarkdown } from "./chunker.js";
import {
  MEDIA_EXTENSIONS,
  MEDIA_MIME_TYPES,
  type Chunk,
  type EmbeddingClient,
  type IndexManager,
  type MediaModality,
  type StoredFile,
  type SyncOptions,
} from "./types.js";

const DEFAULT_MAX_MEDIA_FILE_BYTES = 10 * 1024 * 1024;

export interface SyncManager {
  sync(opts?: SyncOptions): Promise<void>;
  markDirty(): void;
  isDirty(): boolean;
  startWatching(opts?: {
    debounceMs?: number;
    intervalMinutes?: number;
  }): void;
  warmSession(sessionKey?: string): Promise<void>;
  syncIfDirty(): void;
  close(): void;
}

function toRelPath(baseDir: string, absPath: string): string {
  return path.relative(baseDir, absPath).split(path.sep).join("/");
}

function walkDir(dir: string, baseDir: string, out: string[]): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkDir(abs, baseDir, out);
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      out.push(toRelPath(baseDir, abs));
    }
  }
}

function discoverMemoryFiles(workspaceDir: string): string[] {
  const files: string[] = [];

  if (fs.existsSync(workspaceDir) && fs.statSync(workspaceDir).isDirectory()) {
    const rootEntries = new Set(fs.readdirSync(workspaceDir));
    for (const name of ["MEMORY.md", "memory.md"]) {
      if (rootEntries.has(name)) {
        files.push(name);
      }
    }
  }

  const memoryDir = path.join(workspaceDir, "memory");
  if (fs.existsSync(memoryDir) && fs.statSync(memoryDir).isDirectory()) {
    walkDir(memoryDir, workspaceDir, files);
  }

  return Array.from(new Set(files)).sort((a, b) => a.localeCompare(b));
}

function walkMediaDir(
  dir: string,
  baseDir: string,
  extensions: Set<string>,
  maxFileBytes: number,
  out: string[],
): void {
  const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkMediaDir(abs, baseDir, extensions, maxFileBytes, out);
      continue;
    }

    if (!entry.isFile()) {
      continue;
    }

    const ext = path.extname(entry.name).toLowerCase();
    if (!extensions.has(ext)) {
      continue;
    }

    const stat = fs.statSync(abs);
    if (stat.size > maxFileBytes) {
      continue;
    }

    out.push(toRelPath(baseDir, abs));
  }
}

function discoverMediaFiles(workspaceDir: string, modalities: MediaModality[], maxFileBytes: number): string[] {
  const files: string[] = [];
  const extensions = new Set<string>();

  for (const modality of modalities) {
    for (const ext of MEDIA_EXTENSIONS[modality] ?? []) {
      extensions.add(ext.toLowerCase());
    }
  }

  if (extensions.size === 0) {
    return files;
  }

  if (fs.existsSync(workspaceDir) && fs.statSync(workspaceDir).isDirectory()) {
    const rootEntries = fs
      .readdirSync(workspaceDir, { withFileTypes: true })
      .sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of rootEntries) {
      if (!entry.isFile()) {
        continue;
      }

      const ext = path.extname(entry.name).toLowerCase();
      if (!extensions.has(ext)) {
        continue;
      }

      const abs = path.join(workspaceDir, entry.name);
      const stat = fs.statSync(abs);
      if (stat.size > maxFileBytes) {
        continue;
      }

      files.push(toRelPath(workspaceDir, abs));
    }
  }

  const memoryDir = path.join(workspaceDir, "memory");
  if (fs.existsSync(memoryDir) && fs.statSync(memoryDir).isDirectory()) {
    walkMediaDir(memoryDir, workspaceDir, extensions, maxFileBytes, files);
  }

  return Array.from(new Set(files)).sort((a, b) => a.localeCompare(b));
}

function discoverSessionFiles(sessionsDir: string): string[] {
  if (!(fs.existsSync(sessionsDir) && fs.statSync(sessionsDir).isDirectory())) {
    return [];
  }

  const files = fs
    .readdirSync(sessionsDir, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".jsonl"))
    .map((entry) => entry.name);

  return files;
}

function hashContent(content: string): string {
  return createHash("sha256").update(content).digest("hex").slice(0, 16);
}

function hashBuffer(content: Buffer): string {
  return createHash("sha256").update(content).digest("hex").slice(0, 16);
}

function mediaLabelPrefix(mimeType: string): string {
  if (mimeType.startsWith("image/")) {
    return "Image";
  }
  if (mimeType.startsWith("audio/")) {
    return "Audio";
  }
  return "Media";
}

function buildWatchPatterns(multimodalEnabled: boolean, modalities: MediaModality[]): string[] {
  const patterns = new Set<string>(["MEMORY.md", "memory.md", "memory/**/*.md"]);
  if (!multimodalEnabled) {
    return Array.from(patterns);
  }

  for (const modality of modalities) {
    for (const ext of MEDIA_EXTENSIONS[modality] ?? []) {
      const normalizedExt = ext.toLowerCase();
      patterns.add(`*${normalizedExt}`);
      patterns.add(`memory/**/*${normalizedExt}`);
    }
  }

  return Array.from(patterns);
}

function flattenSessionContent(raw: unknown): string | null {
  if (typeof raw === "string") {
    const normalized = raw.replace(/\r?\n+/g, " ").trim();
    return normalized.length > 0 ? normalized : null;
  }

  if (!Array.isArray(raw)) {
    return null;
  }

  const parts: string[] = [];
  for (const block of raw) {
    if (typeof block !== "object" || block === null) {
      continue;
    }

    const type = "type" in block ? block.type : undefined;
    const text = "text" in block ? block.text : undefined;
    if (type === "text" && typeof text === "string") {
      const normalized = text.replace(/\r?\n+/g, " ").trim();
      if (normalized.length > 0) {
        parts.push(normalized);
      }
    }
  }

  if (parts.length === 0) {
    return null;
  }

  return parts.join(" ");
}

export function flattenSessionJsonl(content: string): { text: string; lineMap: number[] } | null {
  if (content.trim().length === 0) {
    return null;
  }

  const flattenedLines: string[] = [];
  const lineMap: number[] = [];
  const lines = content.split(/\r?\n/);

  for (let i = 0; i < lines.length; i += 1) {
    const rawLine = lines[i];
    if (rawLine.trim().length === 0) {
      continue;
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(rawLine);
    } catch {
      continue;
    }

    if (typeof parsed !== "object" || parsed === null) {
      continue;
    }

    // Support both formats:
    // 1. { role: "user", content: ... }  (flat format)
    // 2. { type: "message", message: { role: "user", content: ... } }  (OpenClaw session format)
    const inner =
      "message" in parsed && typeof (parsed as any).message === "object" && (parsed as any).message !== null
        ? (parsed as any).message
        : parsed;

    const role = "role" in inner ? (inner as any).role : undefined;
    if (role === "system") {
      continue;
    }

    let prefix: string | null = null;
    if (role === "user") {
      prefix = "User: ";
    } else if (role === "assistant") {
      prefix = "Assistant: ";
    }

    if (prefix === null) {
      continue;
    }

    const rawMessage = "content" in inner ? (inner as any).content : undefined;
    const message = flattenSessionContent(rawMessage);
    if (message === null) {
      continue;
    }

    flattenedLines.push(`${prefix}${message}`);
    lineMap.push(i + 1);
  }

  if (flattenedLines.length === 0) {
    return null;
  }

  return {
    text: flattenedLines.join("\n"),
    lineMap,
  };
}

export function remapChunkLines(chunks: Chunk[], lineMap: number[]): void {
  for (const chunk of chunks) {
    const startIdx = chunk.startLine - 1;
    const endIdx = chunk.endLine - 1;
    if (startIdx >= 0 && startIdx < lineMap.length) {
      chunk.startLine = lineMap[startIdx];
    }
    if (endIdx >= 0 && endIdx < lineMap.length) {
      chunk.endLine = lineMap[endIdx];
    }
  }
}

export function createSyncManager(params: {
  workspaceDir: string;
  index: IndexManager;
  embedding: EmbeddingClient;
  chunkTokens: number;
  chunkOverlap: number;
  sessionsDir?: string;
  multimodal?: {
    enabled: boolean;
    modalities: MediaModality[];
    maxFileBytes?: number;
  };
}): SyncManager {
  const { workspaceDir, index, embedding, chunkTokens, chunkOverlap, sessionsDir } = params;
  const multimodalEnabled = params.multimodal?.enabled === true;
  const multimodalModalities = params.multimodal?.modalities ?? [];
  const maxMediaFileBytes = Math.max(
    1,
    Math.trunc(params.multimodal?.maxFileBytes ?? DEFAULT_MAX_MEDIA_FILE_BYTES),
  );
  const trackedFiles = new Set<string>();
  const warmedSessions = new Set<string>();

  let dirty = true;
  let closed = false;
  let watcher: FSWatcher | null = null;
  let sessionWatcher: FSWatcher | null = null;
  let syncTimer: NodeJS.Timeout | null = null;
  let intervalTimer: NodeJS.Timeout | null = null;
  let debounceMs = 1000;

  function markDirty(): void {
    dirty = true;
  }

  function isDirty(): boolean {
    return dirty;
  }

  function clearSyncTimer(): void {
    if (syncTimer !== null) {
      clearTimeout(syncTimer);
      syncTimer = null;
    }
  }

  function clearIntervalTimer(): void {
    if (intervalTimer !== null) {
      clearInterval(intervalTimer);
      intervalTimer = null;
    }
  }

  function closeWatcher(): void {
    if (watcher !== null) {
      void watcher.close().catch(() => {});
      watcher = null;
    }
    if (sessionWatcher !== null) {
      void sessionWatcher.close().catch(() => {});
      sessionWatcher = null;
    }
  }

  async function sync(opts?: SyncOptions): Promise<void> {
    if (closed) {
      return;
    }

    const progress = opts?.progress;
    const force = opts?.force === true;
    const memoryFiles = discoverMemoryFiles(workspaceDir);
    const mediaFiles =
      multimodalEnabled && multimodalModalities.length > 0
        ? discoverMediaFiles(workspaceDir, multimodalModalities, maxMediaFileBytes)
        : [];
    const sessionFiles = sessionsDir ? discoverSessionFiles(sessionsDir) : [];
    const seenFiles = new Set<string>();

    let completed = 0;
    const total = memoryFiles.length + mediaFiles.length + sessionFiles.length;

    if (total === 0) {
      progress?.({ completed: 0, total: 0, label: "No memory/session/media files found" });
    }

    try {
      for (const relPath of memoryFiles) {
        seenFiles.add(relPath);
        trackedFiles.add(relPath);

        const absPath = path.join(workspaceDir, relPath);
        const content = fs.readFileSync(absPath, "utf8");
        const contentHash = hashContent(content);
        const existingHash = index.getFileHash(relPath);

        if (!force && existingHash === contentHash) {
          completed += 1;
          progress?.({ completed, total, label: `Skipped ${relPath}` });
          continue;
        }

        const chunks = chunkMarkdown(content, {
          maxTokens: chunkTokens,
          overlapRatio: chunkOverlap,
        });
        const texts = chunks.map((chunk) => chunk.text);
        const vectors = texts.length === 0 ? [] : await embedding.embedBatch(texts, "RETRIEVAL_DOCUMENT");

        const file: StoredFile = {
          fileKey: relPath,
          contentHash,
          source: "memory",
          indexedAt: Date.now(),
        };
        index.indexFile(file, chunks, vectors);

        completed += 1;
        progress?.({ completed, total, label: `Indexed ${relPath}` });
      }

      if (mediaFiles.length > 0) {
        if (!embedding.supportsMultimodal || embedding.embedMedia === undefined) {
          console.warn(
            "Engram: multimodal indexing is enabled, but the current embedding client does not support media. Skipping media files.",
          );

          for (const relPath of mediaFiles) {
            seenFiles.add(relPath);
            trackedFiles.add(relPath);
            completed += 1;
            progress?.({ completed, total, label: `Skipped ${relPath} (multimodal unsupported)` });
          }
        } else {
          for (const relPath of mediaFiles) {
            seenFiles.add(relPath);
            trackedFiles.add(relPath);

            const absPath = path.join(workspaceDir, relPath);
            const ext = path.extname(relPath).toLowerCase();
            const mimeType = MEDIA_MIME_TYPES[ext];
            if (!mimeType) {
              completed += 1;
              progress?.({ completed, total, label: `Skipped ${relPath} (unknown mime type)` });
              continue;
            }

            // Enforce size limits before reading file contents.
            const stat = fs.statSync(absPath);
            if (stat.size > maxMediaFileBytes) {
              completed += 1;
              progress?.({ completed, total, label: `Skipped ${relPath} (file too large)` });
              continue;
            }

            const content = fs.readFileSync(absPath);
            const contentHash = hashBuffer(content);
            const existingHash = index.getFileHash(relPath);
            if (!force && existingHash === contentHash) {
              completed += 1;
              progress?.({ completed, total, label: `Skipped ${relPath}` });
              continue;
            }

            const vector = await embedding.embedMedia(content, mimeType);
            const prefix = mediaLabelPrefix(mimeType);
            const chunks: Chunk[] = [
              {
                text: `${prefix} file: ${relPath}`,
                startLine: 1,
                endLine: 1,
                hash: contentHash,
              },
            ];

            const file: StoredFile = {
              fileKey: relPath,
              contentHash,
              source: "memory",
              indexedAt: Date.now(),
            };
            index.indexFile(file, chunks, [vector]);

            completed += 1;
            progress?.({ completed, total, label: `Indexed ${relPath}` });
          }
        }
      }

      for (const sessionName of sessionFiles) {
        const relPath = path.posix.join("sessions", sessionName);
        seenFiles.add(relPath);
        trackedFiles.add(relPath);

        const absPath = path.join(sessionsDir ?? "", sessionName);
        const content = fs.readFileSync(absPath, "utf8");
        const contentHash = hashContent(content);
        const existingHash = index.getFileHash(relPath);

        if (!force && existingHash === contentHash) {
          completed += 1;
          progress?.({ completed, total, label: `Skipped ${relPath}` });
          continue;
        }

        const flattened = flattenSessionJsonl(content);
        let chunks: Chunk[] = [];
        if (flattened !== null) {
          chunks = chunkMarkdown(flattened.text, {
            maxTokens: chunkTokens,
            overlapRatio: chunkOverlap,
          });
          remapChunkLines(chunks, flattened.lineMap);
        }

        const texts = chunks.map((chunk) => chunk.text);
        const vectors = texts.length === 0 ? [] : await embedding.embedBatch(texts, "RETRIEVAL_DOCUMENT");

        const file: StoredFile = {
          fileKey: relPath,
          contentHash,
          source: "sessions",
          indexedAt: Date.now(),
        };
        index.indexFile(file, chunks, vectors);

        completed += 1;
        progress?.({ completed, total, label: `Indexed ${relPath}` });
      }

      for (const relPath of Array.from(trackedFiles)) {
        if (seenFiles.has(relPath)) {
          continue;
        }
        index.removeFile(relPath);
        trackedFiles.delete(relPath);
      }

      dirty = false;
    } catch (error) {
      dirty = true;
      throw error;
    }
  }

  function scheduleDebouncedSync(): void {
    clearSyncTimer();
    syncTimer = setTimeout(() => {
      void sync({ reason: "file-change" }).catch(() => {});
    }, debounceMs);
  }

  return {
    sync,

    markDirty,

    isDirty,

    startWatching(opts?: { debounceMs?: number; intervalMinutes?: number }): void {
      if (closed) {
        return;
      }

      closeWatcher();
      clearSyncTimer();
      clearIntervalTimer();

      debounceMs = Math.max(0, Math.trunc(opts?.debounceMs ?? 1000));
      const intervalMinutes = Math.max(0, Math.trunc(opts?.intervalMinutes ?? 5));
      const watchPatterns = buildWatchPatterns(multimodalEnabled, multimodalModalities);

      watcher = watch(watchPatterns, {
        cwd: workspaceDir,
        ignoreInitial: true,
        awaitWriteFinish: {
          stabilityThreshold: debounceMs,
          pollInterval: 100,
        },
      });

      const onFsUpdate = (): void => {
        markDirty();
        scheduleDebouncedSync();
      };

      watcher.on("add", onFsUpdate);
      watcher.on("change", onFsUpdate);
      watcher.on("unlink", onFsUpdate);

      if (sessionsDir && fs.existsSync(sessionsDir)) {
        sessionWatcher = watch("*.jsonl", {
          cwd: sessionsDir,
          ignoreInitial: true,
          awaitWriteFinish: {
            stabilityThreshold: debounceMs,
            pollInterval: 100,
          },
        });

        sessionWatcher.on("add", onFsUpdate);
        sessionWatcher.on("change", onFsUpdate);
        sessionWatcher.on("unlink", onFsUpdate);
      }

      if (intervalMinutes > 0) {
        intervalTimer = setInterval(() => {
          if (isDirty()) {
            void sync({ reason: "interval" }).catch(() => {});
          }
        }, intervalMinutes * 60 * 1000);
      }
    },

    async warmSession(sessionKey?: string): Promise<void> {
      if (closed) {
        return;
      }

      const key = sessionKey ?? "__default__";
      if (warmedSessions.has(key)) {
        return;
      }
      warmedSessions.add(key);

      if (isDirty()) {
        await sync({ reason: `warm-session:${key}` });
      }
    },

    syncIfDirty(): void {
      if (closed) {
        return;
      }
      if (isDirty()) {
        void sync({ reason: "on-search" }).catch(() => {});
      }
    },

    close(): void {
      closed = true;
      closeWatcher();
      clearSyncTimer();
      clearIntervalTimer();
      trackedFiles.clear();
      warmedSessions.clear();
    },
  };
}
