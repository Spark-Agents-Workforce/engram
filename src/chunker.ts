import { createHash } from "node:crypto";
import type { Chunk, ChunkerOptions } from "./types.js";

interface LineInfo {
  text: string;
  start: number;
  end: number;
  originalLine: number;
}

interface Range {
  start: number;
  end: number;
}

interface HeadingEvent {
  index: number;
  level: number;
  label: string;
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function splitLines(text: string, startLine: number): LineInfo[] {
  if (text.length === 0) {
    return [];
  }

  const lines: LineInfo[] = [];
  let cursor = 0;
  let line = startLine;

  while (cursor < text.length) {
    const newlineIndex = text.indexOf("\n", cursor);
    if (newlineIndex === -1) {
      lines.push({
        text: text.slice(cursor),
        start: cursor,
        end: text.length,
        originalLine: line,
      });
      break;
    }

    const end = newlineIndex + 1;
    lines.push({
      text: text.slice(cursor, end),
      start: cursor,
      end,
      originalLine: line,
    });
    cursor = end;
    line += 1;
  }

  return lines;
}

function stripLineTerminator(line: string): string {
  return line.replace(/\r?\n$/, "");
}

function stripFrontmatter(content: string): { body: string; startLine: number } {
  if (!(content.startsWith("---\n") || content.startsWith("---\r\n"))) {
    return { body: content, startLine: 1 };
  }

  const lines = splitLines(content, 1);
  if (lines.length === 0 || stripLineTerminator(lines[0].text) !== "---") {
    return { body: content, startLine: 1 };
  }

  for (let i = 1; i < lines.length; i += 1) {
    if (stripLineTerminator(lines[i].text) === "---") {
      return {
        body: content.slice(lines[i].end),
        startLine: lines[i].originalLine + 1,
      };
    }
  }

  return { body: content, startLine: 1 };
}

function hashText(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

function boundaryAtOrBefore(boundaries: number[], minExclusive: number, maxInclusive: number): number | undefined {
  let low = 0;
  let high = boundaries.length - 1;
  let candidate = -1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (boundaries[mid] <= maxInclusive) {
      candidate = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  while (candidate >= 0) {
    const value = boundaries[candidate];
    if (value > minExclusive) {
      return value;
    }
    candidate -= 1;
  }

  return undefined;
}

function boundaryAtOrAfter(boundaries: number[], minInclusive: number, maxInclusive: number): number | undefined {
  let low = 0;
  let high = boundaries.length - 1;
  let candidate = -1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (boundaries[mid] >= minInclusive) {
      candidate = mid;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }

  if (candidate === -1) {
    return undefined;
  }

  const value = boundaries[candidate];
  return value <= maxInclusive ? value : undefined;
}

function rangeContaining(ranges: Range[], position: number): Range | undefined {
  for (const range of ranges) {
    if (position < range.start) {
      return undefined;
    }
    if (position >= range.start && position < range.end) {
      return range;
    }
  }
  return undefined;
}

function normalizeLimitForCode(ranges: Range[], start: number, proposedLimit: number): number {
  const currentRange = rangeContaining(ranges, start);
  if (currentRange) {
    return currentRange.end;
  }

  for (const range of ranges) {
    if (range.start >= proposedLimit) {
      break;
    }
    if (proposedLimit > range.start && proposedLimit < range.end) {
      if (range.start > start) {
        return range.start;
      }
      return range.end;
    }
  }

  return proposedLimit;
}

function lineForOffset(lines: LineInfo[], offset: number): number {
  if (lines.length === 0) {
    return 1;
  }

  let low = 0;
  let high = lines.length - 1;
  let index = 0;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    if (lines[mid].start <= offset) {
      index = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  if (offset >= lines[index].end && index < lines.length - 1) {
    return lines[index + 1].originalLine;
  }

  return lines[index].originalLine;
}

function headingContextAt(headings: HeadingEvent[], offset: number): string | undefined {
  const stack: string[] = [];

  for (const heading of headings) {
    if (heading.index > offset) {
      break;
    }
    stack.length = heading.level - 1;
    stack[heading.level - 1] = heading.label;
  }

  return stack.length > 0 ? stack.join(" > ") : undefined;
}

function nearestBoundary(
  boundaries: number[],
  minInclusive: number,
  maxInclusive: number,
  target: number,
): number | undefined {
  if (minInclusive > maxInclusive) {
    return undefined;
  }

  const before = boundaryAtOrBefore(boundaries, minInclusive - 1, target);
  const after = boundaryAtOrAfter(boundaries, target, maxInclusive);

  if (before === undefined) {
    return after;
  }
  if (after === undefined) {
    return before;
  }

  return Math.abs(after - target) <= Math.abs(target - before) ? after : before;
}

export function chunkMarkdown(content: string, options: ChunkerOptions): Chunk[] {
  if (content.length === 0) {
    return [];
  }

  const maxTokens = Math.max(1, options.maxTokens);
  const maxChars = maxTokens * 4;
  const overlapTokens = Math.max(0, Math.floor(maxTokens * options.overlapRatio));
  const overlapChars = overlapTokens * 4;

  const stripped = stripFrontmatter(content);
  const body = stripped.body;

  if (body.length === 0 || body.trim().length === 0) {
    return [];
  }

  const lines = splitLines(body, stripped.startLine);
  if (lines.length === 0) {
    return [];
  }

  const headingBoundaries: number[] = [];
  const blankLineBoundaries: number[] = [];
  const lineBoundaries: number[] = [0];
  const headingEvents: HeadingEvent[] = [];
  const codeRanges: Range[] = [];

  let inCodeBlock = false;
  let codeStart = 0;

  for (const line of lines) {
    lineBoundaries.push(line.start);
    const withoutTerminator = stripLineTerminator(line.text);
    const isFence = /^\s*```/.test(withoutTerminator);

    if (isFence) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeStart = line.start;
      } else {
        inCodeBlock = false;
        codeRanges.push({ start: codeStart, end: line.end });
      }
      continue;
    }

    if (inCodeBlock) {
      continue;
    }

    const headingMatch = withoutTerminator.match(/^(#{1,6})\s+(.+?)\s*$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      headingBoundaries.push(line.start);
      headingEvents.push({
        index: line.start,
        level,
        label: `${"#".repeat(level)} ${headingMatch[2].trim()}`,
      });
    }

    if (withoutTerminator.trim().length === 0) {
      blankLineBoundaries.push(line.end);
    }
  }

  if (inCodeBlock) {
    codeRanges.push({ start: codeStart, end: body.length });
  }

  const sentenceBoundaries: number[] = [];
  const wordBoundaries: number[] = [];
  let codeIndex = 0;

  for (let i = 0; i < body.length; i += 1) {
    while (codeIndex < codeRanges.length && i >= codeRanges[codeIndex].end) {
      codeIndex += 1;
    }

    const inCode =
      codeIndex < codeRanges.length && i >= codeRanges[codeIndex].start && i < codeRanges[codeIndex].end;
    if (inCode) {
      continue;
    }

    const char = body[i];
    if (/\s/.test(char)) {
      wordBoundaries.push(i + 1);
    }

    if (char === "." || char === "?" || char === "!") {
      const next = body[i + 1];
      if (next === undefined || /\s/.test(next)) {
        sentenceBoundaries.push(i + 1);
      }
    }
  }

  const chunks: Chunk[] = [];
  let start = 0;

  while (start < body.length) {
    const remaining = body.slice(start);
    if (estimateTokens(remaining) <= maxTokens) {
      const chunkText = remaining;
      const startLine = lineForOffset(lines, start);
      const endLine = lineForOffset(lines, body.length - 1);
      chunks.push({
        text: chunkText,
        startLine,
        endLine,
        hash: hashText(chunkText),
        headingContext: headingContextAt(headingEvents, start),
      });
      break;
    }

    const rawLimit = Math.min(body.length, start + maxChars);
    const normalizedLimit = normalizeLimitForCode(codeRanges, start, rawLimit);

    let split =
      boundaryAtOrBefore(headingBoundaries, start, normalizedLimit) ??
      boundaryAtOrBefore(blankLineBoundaries, start, normalizedLimit) ??
      boundaryAtOrBefore(sentenceBoundaries, start, normalizedLimit) ??
      boundaryAtOrBefore(wordBoundaries, start, normalizedLimit) ??
      normalizedLimit;

    if (split <= start) {
      const containingCode = rangeContaining(codeRanges, start);
      split = containingCode ? containingCode.end : Math.min(body.length, start + maxChars);
      if (split <= start) {
        split = Math.min(body.length, start + 1);
      }
    }

    const chunkText = body.slice(start, split);
    const startLine = lineForOffset(lines, start);
    const endLine = lineForOffset(lines, split - 1);
    chunks.push({
      text: chunkText,
      startLine,
      endLine,
      hash: hashText(chunkText),
      headingContext: headingContextAt(headingEvents, start),
    });

    if (split >= body.length) {
      break;
    }

    if (overlapChars <= 0) {
      start = split;
      continue;
    }

    const minStart = start + 1;
    const maxStart = split - 1;
    if (minStart > maxStart) {
      start = split;
      continue;
    }

    const target = Math.max(minStart, Math.min(maxStart, split - overlapChars));
    const overlapCode = rangeContaining(codeRanges, target);

    let nextStart: number;
    if (overlapCode) {
      nextStart = overlapCode.start;
    } else {
      nextStart =
        nearestBoundary(sentenceBoundaries, minStart, maxStart, target) ??
        nearestBoundary(lineBoundaries, minStart, maxStart, target) ??
        target;
    }

    const nextStartCode = rangeContaining(codeRanges, nextStart);
    if (nextStartCode) {
      nextStart = nextStartCode.start > start ? nextStartCode.start : split;
    }

    if (nextStart <= start || nextStart >= split) {
      start = split;
    } else {
      start = nextStart;
    }
  }

  return chunks;
}
