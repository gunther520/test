import { fetchWithTimeout, ProviderError, unwrapDuckDuckGoUrl, decodeHtml } from "../http";
import { FILTERABLE_PLATFORMS, giveawayQuery, siteQueryFor } from "../platforms";
import { normalizeHit, type RawHit } from "../normalize";
import type { Giveaway, PlatformFilter } from "../types";

const RSS_PER_FEED = 12;

function parseDuckDuckGo(html: string): RawHit[] {
  const hits: RawHit[] = [];
  const blockRe =
    /<div class="result[^"]*"[\s\S]*?<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?(?:<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>|<td class="result-snippet">([\s\S]*?)<\/td>)?/gi;

  let match: RegExpExecArray | null;
  while ((match = blockRe.exec(html))) {
    const url = unwrapDuckDuckGoUrl(decodeHtml(match[1]));
    const title = decodeHtml(match[2]);
    const snippet = decodeHtml(match[3] || match[4] || "");
    if (!url || !title) continue;
    if (url.includes("duckduckgo.com") && !url.includes("uddg=")) continue;
    hits.push({ title, url, snippet, source: "web-search" });
  }

  if (hits.length === 0) {
    const liteRe =
      /<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
    while ((match = liteRe.exec(html))) {
      const url = unwrapDuckDuckGoUrl(decodeHtml(match[1]));
      const title = decodeHtml(match[2]);
      if (!url.startsWith("http") || url.includes("duckduckgo.com")) continue;
      hits.push({ title, url, snippet: "", source: "web-search" });
    }
  }

  return hits;
}

async function searchDuckDuckGo(query: string): Promise<RawHit[]> {
  const params = new URLSearchParams({ q: query, kl: "us-en" });
  const url = `https://html.duckduckgo.com/html/?${params.toString()}`;
  const response = await fetchWithTimeout(url, {
    headers: {
      accept: "text/html",
      referer: "https://html.duckduckgo.com/",
    },
  });
  if (!response.ok) {
    throw new ProviderError(`DuckDuckGo returned HTTP ${response.status}`, response.status);
  }
  const html = await response.text();
  if (/anomaly-modal|captcha|bot.?detect/i.test(html) && !/result__a/.test(html)) {
    throw new ProviderError(
      "DuckDuckGo presented a bot check; falling back to other public sources",
    );
  }
  return parseDuckDuckGo(html);
}

function parseRss(xml: string): RawHit[] {
  const items: RawHit[] = [];
  const itemRe = /<item>([\s\S]*?)<\/item>/gi;
  let block: RegExpExecArray | null;
  while ((block = itemRe.exec(xml))) {
    const chunk = block[1];
    const title = decodeHtml(chunk.match(/<title>([\s\S]*?)<\/title>/i)?.[1] ?? "");
    const link = decodeHtml(
      chunk.match(/<link>([\s\S]*?)<\/link>/i)?.[1] ??
        chunk.match(/<guid[^>]*>([\s\S]*?)<\/guid>/i)?.[1] ??
        "",
    );
    const snippet = decodeHtml(
      chunk.match(/<description>([\s\S]*?)<\/description>/i)?.[1] ?? "",
    );
    const sourceName = decodeHtml(
      chunk.match(/<source[^>]*>([\s\S]*?)<\/source>/i)?.[1] ?? "",
    );
    const pub = chunk.match(/<pubDate>([\s\S]*?)<\/pubDate>/i)?.[1];
    let publishedAt: string | undefined;
    if (pub) {
      const date = new Date(decodeHtml(pub));
      if (!Number.isNaN(date.getTime())) publishedAt = date.toISOString();
    }
    if (title && link) {
      items.push({
        title,
        url: link,
        snippet: [sourceName, snippet].filter(Boolean).join(" — ").slice(0, 280),
        publishedAt,
        source: "web-search",
      });
    }
    if (items.length >= RSS_PER_FEED) break;
  }
  return items;
}

async function searchNewsRss(query: string): Promise<RawHit[]> {
  const params = new URLSearchParams({
    q: query,
    hl: "en-US",
    gl: "US",
    ceid: "US:en",
  });
  const url = `https://news.google.com/rss/search?${params.toString()}`;
  const response = await fetchWithTimeout(url, {
    headers: { accept: "application/rss+xml, application/xml, text/xml" },
  });
  if (!response.ok) {
    throw new ProviderError(`Google News RSS returned HTTP ${response.status}`, response.status);
  }
  return parseRss(await response.text());
}

function rssQueries(userQuery: string, platform: PlatformFilter): string[] {
  const base = giveawayQuery(userQuery);
  if (platform !== "all") {
    return [`${base} ${siteQueryFor(platform)}`];
  }
  return FILTERABLE_PLATFORMS.filter((item) => item.id !== "reddit").map(
    (item) => `${base} ${item.siteQuery}`,
  );
}

export async function searchWeb(
  userQuery: string,
  platform: PlatformFilter,
): Promise<Giveaway[]> {
  const q = `${giveawayQuery(userQuery)} (${siteQueryFor(platform)})`;
  const errors: string[] = [];
  let hits: RawHit[] = [];

  try {
    hits = await searchDuckDuckGo(q);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "DuckDuckGo failed");
  }

  try {
    const rssGroups = await Promise.allSettled(
      rssQueries(userQuery, platform).map((query) => searchNewsRss(query)),
    );
    const rssHits: RawHit[] = [];
    let rssFailed = 0;
    for (const group of rssGroups) {
      if (group.status === "fulfilled") rssHits.push(...group.value);
      else rssFailed += 1;
    }
    hits = hits.concat(rssHits);
    if (rssFailed === rssGroups.length && rssGroups.length > 0) {
      errors.push("Google News RSS failed");
    }
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "News RSS failed");
  }

  const results = hits
    .map((hit) => normalizeHit(hit, platform))
    .filter((item): item is Giveaway => item !== null);

  if (results.length === 0 && errors.length > 0 && hits.length === 0) {
    throw new ProviderError(errors.join("; "));
  }
  return results;
}
