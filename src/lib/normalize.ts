import { extractDates, extractUrlDate } from "./dates";
import { classifyPlatform } from "./platforms";
import type { Giveaway, PlatformFilter, ProviderId } from "./types";
import { looksLikeGiveaway, stableId } from "./http";

export type RawHit = {
  title: string;
  url: string;
  snippet: string;
  publishedAt?: string;
  source: ProviderId;
};

export function normalizeHit(
  hit: RawHit,
  filter: PlatformFilter,
): Giveaway | null {
  if (!hit.url || !hit.title) return null;
  let url = hit.url.trim();
  if (!/^https?:\/\//i.test(url)) {
    url = `https://${url}`;
  }
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return null;
    }
    url = parsed.toString();
  } catch {
    return null;
  }

  const platform = classifyPlatform(url, `${hit.title} ${hit.snippet}`);
  if (filter !== "all" && platform !== filter) return null;
  const social = platform !== "other";
  if (!looksLikeGiveaway(hit.title, hit.snippet) && !social) {
    return null;
  }

  const dates = extractDates(`${hit.title} ${hit.snippet}`);
  return {
    id: stableId(url),
    title: hit.title.trim().slice(0, 180),
    platform,
    url,
    snippet: hit.snippet.trim().slice(0, 280),
    publishedAt: hit.publishedAt ?? dates.publishedAt ?? extractUrlDate(url),
    endsAt: dates.endsAt,
    source: hit.source,
  };
}

export function dedupe(results: Giveaway[]): Giveaway[] {
  const seen = new Set<string>();
  const out: Giveaway[] = [];
  for (const item of results) {
    const key = item.url.replace(/\/+$/, "").toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}
