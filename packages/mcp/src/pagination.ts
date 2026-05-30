// Opaque base64 cursor helpers per ARCHITECTURE.md §11.
// Cursor encodes { offset, filter_hash } so we can reject reuse across different filter sets.

import { createHash } from "node:crypto";

export interface CursorPayload {
  offset: number;
  filter_hash: string;
}

export function hashFilters(filters: unknown): string {
  const json = JSON.stringify(filters ?? {});
  return createHash("sha256").update(json).digest("hex").slice(0, 12);
}

export function encodeCursor(payload: CursorPayload): string {
  const json = JSON.stringify(payload);
  return Buffer.from(json, "utf8").toString("base64url");
}

export function decodeCursor(token: string): CursorPayload | null {
  try {
    const json = Buffer.from(token, "base64url").toString("utf8");
    const obj = JSON.parse(json);
    if (
      obj &&
      typeof obj === "object" &&
      typeof obj.offset === "number" &&
      typeof obj.filter_hash === "string"
    ) {
      return { offset: obj.offset, filter_hash: obj.filter_hash };
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Paginate an in-memory array. Validates cursor against the current filter_hash.
 * Returns { page, next_cursor }.
 */
export function paginate<T>(
  items: T[],
  cursorToken: string | undefined,
  limit: number,
  filters: unknown,
): { ok: true; page: T[]; next_cursor?: string } | { ok: false; reason: "invalid_cursor" } {
  const filter_hash = hashFilters(filters);
  let offset = 0;
  if (cursorToken) {
    const decoded = decodeCursor(cursorToken);
    if (!decoded || decoded.filter_hash !== filter_hash) {
      return { ok: false, reason: "invalid_cursor" };
    }
    offset = decoded.offset;
  }
  const page = items.slice(offset, offset + limit);
  const newOffset = offset + page.length;
  const more = newOffset < items.length;
  return {
    ok: true,
    page,
    next_cursor: more ? encodeCursor({ offset: newOffset, filter_hash }) : undefined,
  };
}
