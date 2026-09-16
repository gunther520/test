import { isRecentGiveaway, recencyTimestamp } from "./dates";
import { searchReddit } from "./providers/reddit";
import { searchWeb } from "./providers/web-search";
import { dedupe } from "./normalize";
import { parsePlatformFilter } from "./platforms";
import { isRelevant, relevanceScore } from "./score";
import type {
  Giveaway,
  PlatformFilter,
  ProviderStatus,
  SearchResponse,
} from "./types";

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
    runProvider("reddit", "Reddit public JSON", false, true, true, () =>
      searchReddit(query, platform),
    ),
  ]);

  const results = dedupe(jobs.flatMap((job) => job.results))
    .filter(isRelevant)
    .filter((item) => isRecentGiveaway(item))
    .sort((a, b) => {
      const qualityDiff = relevanceScore(b) - relevanceScore(a);
      if (qualityDiff !== 0) return qualityDiff;
      return recencyTimestamp(b) - recencyTimestamp(a);
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
      "Public search sources did not respond. Drawboard does not log into social apps or use paid search APIs — retry or try another query.",
    );
  }

  return {
    query,
    platform,
    results,
    providers,
    warnings,
    setup: [],
  };
}

export function statusPayload() {
  return {
    setup: [],
    note: "Free-tier public search only: DuckDuckGo HTML when available, otherwise Google News RSS with site: patterns, plus Reddit public JSON when the host allows it. No API keys required. Results are limited to the last 31 days.",
  };
}
