import { braveConfigured, searchBrave } from "./providers/brave";
import { googleCseConfigured, searchGoogleCse } from "./providers/google-cse";
import { searchReddit } from "./providers/reddit";
import { searchWeb } from "./providers/web-search";
import { searchYoutube, youtubeConfigured } from "./providers/youtube";
import { dedupe } from "./normalize";
import { parsePlatformFilter } from "./platforms";
import { isRelevant, relevanceScore } from "./score";
import type {
  Giveaway,
  PlatformFilter,
  ProviderStatus,
  SearchResponse,
  SetupHint,
} from "./types";

export function setupHints(): SetupHint[] {
  return [
    {
      env: "YOUTUBE_API_KEY",
      purpose: "Official YouTube Data API search for giveaway videos",
      docs: "https://developers.google.com/youtube/v3/getting-started",
      set: youtubeConfigured(),
    },
    {
      env: "BRAVE_SEARCH_API_KEY",
      purpose: "Brave Search API (preferred public web results)",
      docs: "https://brave.com/search/api/",
      set: braveConfigured(),
    },
    {
      env: "GOOGLE_API_KEY + GOOGLE_CSE_ID",
      purpose: "Google Programmable Search Engine for indexed social URLs",
      docs: "https://programmablesearchengine.google.com/",
      set: googleCseConfigured(),
    },
  ];
}

async function runProvider(
  id: string,
  label: string,
  optional: boolean,
  configured: boolean,
  used: boolean,
  task: () => Promise<Giveaway[]>,
): Promise<{ results: Giveaway[]; status: ProviderStatus }> {
  if (!configured || !used) {
    return {
      results: [],
      status: {
        id,
        label,
        configured,
        used: false,
        ok: true,
        optional,
        count: 0,
        hint: configured
          ? undefined
          : optional
            ? `Not configured — set ${label} keys in .env.local`
            : undefined,
      },
    };
  }

  try {
    const results = await task();
    return {
      results,
      status: {
        id,
        label,
        configured,
        used: true,
        ok: true,
        optional,
        count: results.length,
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return {
      results: [],
      status: {
        id,
        label,
        configured,
        used: true,
        ok: false,
        optional,
        count: 0,
        error: message,
      },
    };
  }
}

export async function searchGiveaways(
  rawQuery: string,
  rawPlatform: string | null,
): Promise<SearchResponse> {
  const query = rawQuery.trim().slice(0, 120) || "giveaway";
  const platform: PlatformFilter = parsePlatformFilter(rawPlatform);
  const warnings: string[] = [];

  const jobs = await Promise.all([
    runProvider("web-search", "Public web search", false, true, true, () =>
      searchWeb(query, platform),
    ),
    runProvider(
      "brave",
      "Brave Search",
      true,
      braveConfigured(),
      braveConfigured(),
      () => searchBrave(query, platform),
    ),
    runProvider(
      "google-cse",
      "Google Programmable Search",
      true,
      googleCseConfigured(),
      googleCseConfigured(),
      () => searchGoogleCse(query, platform),
    ),
    runProvider("reddit", "Reddit public JSON", false, true, true, () =>
      searchReddit(query, platform),
    ),
    runProvider(
      "youtube",
      "YouTube Data API",
      true,
      youtubeConfigured(),
      youtubeConfigured() && (platform === "all" || platform === "youtube"),
      () => searchYoutube(query, platform),
    ),
  ]);

  const results = dedupe(jobs.flatMap((job) => job.results))
    .filter(isRelevant)
    .sort((a, b) => {
      const scoreDiff = relevanceScore(b) - relevanceScore(a);
      if (scoreDiff !== 0) return scoreDiff;
      const aTime = a.publishedAt ? Date.parse(a.publishedAt) : 0;
      const bTime = b.publishedAt ? Date.parse(b.publishedAt) : 0;
      return bTime - aTime;
    })
    .slice(0, 24);

  const providers = jobs.map((job) => job.status);
  for (const status of providers) {
    if (status.used && !status.ok && status.error) {
      if (status.id === "reddit" && results.length > 0) continue;
      warnings.push(`${status.label}: ${status.error}`);
    }
  }

  const anyOk = providers.some((p) => p.used && p.ok);
  if (!anyOk) {
    warnings.unshift(
      "Every search backend failed. The tracker does not log into social apps or bypass access controls — add an API key or retry.",
    );
  }

  return {
    query,
    platform,
    results,
    providers,
    warnings,
    setup: setupHints(),
  };
}

export function statusPayload() {
  return {
    setup: setupHints(),
    note: "Works without keys via public web search (DuckDuckGo when available, otherwise Google News RSS) plus Reddit's public JSON API. Official YouTube / Brave / Google CSE APIs are used when configured.",
  };
}
