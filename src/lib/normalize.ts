import { extractDates, extractUrlDate } from "./dates";
import { classifyPlatform } from "./platforms";
import { isEnterableGiveaway, isJunkNewsHost } from "./quality";
import type { Giveaway, PlatformFilter, ProviderId } from "./types";
import { stableId } from "./http";

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
  if (isJunkNewsHost(url)) return null;
  const dates = extractDates(`${hit.title} ${hit.snippet}`);
  const candidate = {
    id: stableId(url),
    title: hit.title.trim().slice(0, 180),
    platform,
    url,
    snippet: hit.snippet.trim().slice(0, 280),
    publishedAt: hit.publishedAt ?? dates.publishedAt ?? extractUrlDate(url),
    endsAt: dates.endsAt,
    source: hit.source,
  };
  if (!isEnterableGiveaway(candidate)) return null;
  return candidate;
}

export function canonicalUrl(url: string): string {
  return url.replace(/\/+$/, "").toLowerCase();
}

export function dedupe(results: Giveaway[]): Giveaway[] {
  const seen = new Set<string>();
  const out: Giveaway[] = [];
  for (const item of results) {
    const key = canonicalUrl(item.url);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out;
}
