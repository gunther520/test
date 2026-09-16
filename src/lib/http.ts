const DEFAULT_UA =
  "GiveawayTracker/0.1 (+https://github.com/gunther520/test; public-search MVP)";

export class ProviderError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ProviderError";
  }
}

export async function fetchWithTimeout(
  url: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<Response> {
  const { timeoutMs = 8000, headers, ...rest } = init;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...rest,
      signal: controller.signal,
      headers: {
        "user-agent": DEFAULT_UA,
        accept: "text/html,application/json,application/rss+xml,application/xml;q=0.9,*/*;q=0.8",
        ...headers,
      },
      redirect: "follow",
      cache: "no-store",
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new ProviderError("Request timed out");
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export function decodeHtml(value: string): string {
  return value
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/\s+/g, " ")
    .trim();
}

export function stableId(url: string): string {
  let hash = 0;
  for (let i = 0; i < url.length; i += 1) {
    hash = (hash * 31 + url.charCodeAt(i)) >>> 0;
  }
  return hash.toString(16);
}

export function unwrapDuckDuckGoUrl(href: string): string {
  try {
    const absolute = href.startsWith("//")
      ? `https:${href}`
      : href.startsWith("http")
        ? href
        : `https://html.duckduckgo.com${href}`;
    const parsed = new URL(absolute);
    const uddg = parsed.searchParams.get("uddg");
    if (uddg) return uddg;
    return parsed.toString();
  } catch {
    return href;
  }
}

export function looksLikeGiveaway(title: string, snippet: string): boolean {
  const text = `${title} ${snippet}`.toLowerCase();
  return /giveaway|give away|lucky draw|raffle|sweepstake|contest|prize|comment to win|retweet to win|follow to enter|\bwin\b/.test(
    text,
  );
}
