import fs from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import type { Chunk, IndexManager, MemorySource, ScoredChunk, StoredFile } from "./types.js";

interface Bm25Row {
  id: number;
  file_key: string;
  start_line: number;
  end_line: number;
  text: string;
  heading_context: string | null;
  source: MemorySource;
  indexed_at: number;
  rank: number;
}

interface VectorHitRow {
  chunk_id: number;
  distance: number;
}

interface ChunkRow {
  id: number;
  file_key: string;
  start_line: number;
  end_line: number;
  text: string;
  heading_context: string | null;
  source: MemorySource;
}

interface FileIndexedAtRow {
  indexed_at: number;
}

interface CountRow {
  count: number;
}

interface SourceStatsRow {
  source: MemorySource;
  files: number;
  chunks: number;
}

function toVectorBlob(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

function toScoredChunk(chunk: ChunkRow, score: number, indexedAt?: number): ScoredChunk {
  return {
    fileKey: chunk.file_key,
    startLine: chunk.start_line,
    endLine: chunk.end_line,
    text: chunk.text,
    score,
    source: chunk.source,
    headingContext: chunk.heading_context ?? undefined,
    indexedAt,
  };
}

export function createIndexManager(params: {
  dbPath: string;
  dimensions: number;
  workspaceDir: string;
}): IndexManager {
  const { dbPath, dimensions, workspaceDir } = params;

  if (!Number.isInteger(dimensions) || dimensions <= 0) {
    throw new Error(`Invalid vector dimensions: ${dimensions}`);
  }

  if (dbPath !== ":memory:") {
    fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  }

  const db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.pragma("foreign_keys = ON");

  let vectorEnabled = true;
  try {
    sqliteVec.load(db);
  } catch {
    vectorEnabled = false;
  }

  db.exec(`
    CREATE TABLE IF NOT EXISTS files (
      file_key TEXT PRIMARY KEY,
      content_hash TEXT NOT NULL,
      source TEXT NOT NULL,
      indexed_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file_key TEXT NOT NULL,
      start_line INTEGER NOT NULL,
      end_line INTEGER NOT NULL,
      text TEXT NOT NULL,
      heading_context TEXT,
      source TEXT NOT NULL,
      FOREIGN KEY (file_key) REFERENCES files(file_key)
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
      text,
      content='chunks',
      content_rowid='id',
      tokenize='porter unicode61'
    );

    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END;

    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;

    CREATE TABLE IF NOT EXISTS meta (
      key TEXT PRIMARY KEY,
      value TEXT
    );
  `);

  if (vectorEnabled) {
    db.exec(`
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
        chunk_id INTEGER PRIMARY KEY,
        embedding float[${dimensions}]
      );
    `);
  }

  const deleteVectorsByFileStmt = vectorEnabled
    ? db.prepare(
        `DELETE FROM chunks_vec WHERE chunk_id IN (SELECT id FROM chunks WHERE file_key = ?);`,
      )
    : null;
  const deleteChunksByFileStmt = db.prepare(`DELETE FROM chunks WHERE file_key = ?;`);
  const deleteFileStmt = db.prepare(`DELETE FROM files WHERE file_key = ?;`);
  const insertFileStmt = db.prepare(
    `INSERT INTO files (file_key, content_hash, source, indexed_at) VALUES (?, ?, ?, ?);`,
  );
  const insertChunkStmt = db.prepare(
    `INSERT INTO chunks (file_key, start_line, end_line, text, heading_context, source) VALUES (?, ?, ?, ?, ?, ?);`,
  );
  const bm25Stmt = db.prepare<[string, number], Bm25Row>(`
    SELECT
      c.id,
      c.file_key,
      c.start_line,
      c.end_line,
      c.text,
      c.heading_context,
      c.source,
      f.indexed_at,
      rank
    FROM chunks_fts fts
    JOIN chunks c ON c.id = fts.rowid
    JOIN files f ON f.file_key = c.file_key
    WHERE chunks_fts MATCH ?
    ORDER BY rank
    LIMIT ?;
  `);
  const vectorSearchStmt = vectorEnabled
    ? db.prepare<[Buffer, number], VectorHitRow>(`
        SELECT chunk_id, distance
        FROM chunks_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance;
      `)
    : null;
  const getChunkByIdStmt = db.prepare<[number], ChunkRow>(`
    SELECT id, file_key, start_line, end_line, text, heading_context, source
    FROM chunks
    WHERE id = ?;
  `);
  const getFileIndexedAtStmt = db.prepare<[string], FileIndexedAtRow>(
    `SELECT indexed_at FROM files WHERE file_key = ?;`,
  );
  const getFileHashStmt = db.prepare<[string], { content_hash: string }>(
    `SELECT content_hash FROM files WHERE file_key = ?;`,
  );
  const fileCountStmt = db.prepare<[], CountRow>(`SELECT COUNT(*) AS count FROM files;`);
  const chunkCountStmt = db.prepare<[], CountRow>(`SELECT COUNT(*) AS count FROM chunks;`);
  const sourceStatsStmt = db.prepare<[], SourceStatsRow>(`
    SELECT
      s.source AS source,
      COALESCE(f.files, 0) AS files,
      COALESCE(c.chunks, 0) AS chunks
    FROM (
      SELECT 'memory' AS source
      UNION ALL
      SELECT 'sessions' AS source
    ) AS s
    LEFT JOIN (
      SELECT source, COUNT(*) AS files
      FROM files
      GROUP BY source
    ) AS f ON f.source = s.source
    LEFT JOIN (
      SELECT source, COUNT(*) AS chunks
      FROM chunks
      GROUP BY source
    ) AS c ON c.source = s.source;
  `);

  const removeFileTx = db.transaction((fileKey: string) => {
    deleteVectorsByFileStmt?.run(fileKey);
    deleteChunksByFileStmt.run(fileKey);
    deleteFileStmt.run(fileKey);
  });

  const indexFileTx = db.transaction((file: StoredFile, chunks: Chunk[], vectors: Float32Array[]) => {
    if (chunks.length !== vectors.length) {
      throw new Error(
        `Chunk/vector length mismatch for "${file.fileKey}": ${chunks.length} chunks vs ${vectors.length} vectors`,
      );
    }

    deleteVectorsByFileStmt?.run(file.fileKey);
    deleteChunksByFileStmt.run(file.fileKey);
    deleteFileStmt.run(file.fileKey);

    insertFileStmt.run(file.fileKey, file.contentHash, file.source, file.indexedAt);

    for (let i = 0; i < chunks.length; i += 1) {
      const chunk = chunks[i];
      const vector = vectors[i];

      if (vector.length !== dimensions) {
        throw new Error(
          `Vector dimension mismatch at chunk ${i} for "${file.fileKey}": expected ${dimensions}, received ${vector.length}`,
        );
      }

      const insertResult = insertChunkStmt.run(
        file.fileKey,
        chunk.startLine,
        chunk.endLine,
        chunk.text,
        chunk.headingContext ?? null,
        file.source,
      );

      if (vectorEnabled) {
        const chunkId = Number(insertResult.lastInsertRowid);
        const insertVectorStmt = db.prepare(`INSERT INTO chunks_vec (chunk_id, embedding) VALUES (${chunkId}, ?);`);
        insertVectorStmt.run(toVectorBlob(vector));
      }
    }
  });

  return {
    indexFile(file: StoredFile, chunks: Chunk[], vectors: Float32Array[]): void {
      indexFileTx(file, chunks, vectors);
    },

    removeFile(fileKey: string): void {
      removeFileTx(fileKey);
    },

    searchBM25(query: string, topK: number): ScoredChunk[] {
      if (topK <= 0) {
        return [];
      }

      try {
        const rows = bm25Stmt.all(query, topK);
        return rows.map((row) => {
          const score = 1 / (1 + Math.abs(row.rank));
          return toScoredChunk(row, score, row.indexed_at);
        });
      } catch {
        return [];
      }
    },

    searchVector(queryVec: Float32Array, topK: number): ScoredChunk[] {
      if (!vectorEnabled || vectorSearchStmt === null || topK <= 0) {
        return [];
      }

      const hits = vectorSearchStmt.all(toVectorBlob(queryVec), topK);
      if (hits.length === 0) {
        return [];
      }

      const results: ScoredChunk[] = [];
      for (const hit of hits) {
        const chunk = getChunkByIdStmt.get(hit.chunk_id);
        if (!chunk) {
          continue;
        }

        const indexedAt = getFileIndexedAtStmt.get(chunk.file_key)?.indexed_at;
        const score = 1 / (1 + hit.distance);
        results.push(toScoredChunk(chunk, score, indexedAt));
      }
      return results;
    },

    getFileHash(fileKey: string): string | null {
      const row = getFileHashStmt.get(fileKey);
      return row?.content_hash ?? null;
    },

    readFileContent(relPath: string, from?: number, lines?: number): { text: string; path: string } | null {
      const normalizedRelPath = path.posix
        .normalize(relPath.replace(/\\/g, "/").replace(/^\.\/+/, ""))
        .replace(/^\/+/, "");
      if (
        normalizedRelPath.length === 0 ||
        normalizedRelPath === "." ||
        normalizedRelPath === ".." ||
        normalizedRelPath.startsWith("../")
      ) {
        return null;
      }

      const absPath = path.resolve(workspaceDir, normalizedRelPath);
      if (!fs.existsSync(absPath)) {
        return null;
      }

      const realWorkspace = fs.realpathSync(workspaceDir);
      const realAbs = fs.realpathSync(absPath);
      if (realAbs !== realWorkspace && !realAbs.startsWith(`${realWorkspace}${path.sep}`)) {
        return null;
      }

      const text = fs.readFileSync(absPath, "utf8");
      if (from === undefined && lines === undefined) {
        return { text, path: normalizedRelPath };
      }

      const split = text.split(/\r?\n/);
      const start = Math.max((from ?? 1) - 1, 0);
      const count = lines === undefined ? split.length - start : Math.max(lines, 0);
      const slicedText = split.slice(start, start + count).join("\n");
      return { text: slicedText, path: normalizedRelPath };
    },

    stats() {
      const fileCount = fileCountStmt.get()?.count ?? 0;
      const chunkCount = chunkCountStmt.get()?.count ?? 0;
      const sources = sourceStatsStmt.all().map((row) => ({
        source: row.source as MemorySource,
        files: Number(row.files),
        chunks: Number(row.chunks),
      }));

      return {
        files: Number(fileCount),
        chunks: Number(chunkCount),
        sources,
        dbPath,
        vectorDims: vectorEnabled ? dimensions : 0,
      };
    },

    close(): void {
      if (db.open) {
        db.close();
      }
    },
  };
}
