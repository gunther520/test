/** Leftover optional provider. Not invoked by supported free-tier search. */
import { fetchWithTimeout, ProviderError } from "../http";
import { giveawayQuery, siteQueryFor } from "../platforms";
import { normalizeHit } from "../normalize";
import type { Giveaway, PlatformFilter } from "../types";

type BraveWeb = {
  web?: {
    results?: Array<{
      title?: string;
      url?: string;
      description?: string;
    }>;
  };
  message?: string;
};

export function braveConfigured(): boolean {
  return Boolean(process.env.BRAVE_SEARCH_API_KEY);
}

export async function searchBrave(
  userQuery: string,
  platform: PlatformFilter,
): Promise<Giveaway[]> {
  const key = process.env.BRAVE_SEARCH_API_KEY;
  if (!key) return [];

  const q = `${giveawayQuery(userQuery)} (${siteQueryFor(platform)})`;
  const params = new URLSearchParams({ q, count: "15", text_decorations: "false" });
  const response = await fetchWithTimeout(
    `https://api.search.brave.com/res/v1/web/search?${params.toString()}`,
    {
      headers: {
        accept: "application/json",
        "X-Subscription-Token": key,
      },
    },
  );
  const json = (await response.json()) as BraveWeb;
  if (!response.ok) {
    throw new ProviderError(
      json.message ?? `Brave Search returned HTTP ${response.status}`,
      response.status,
    );
  }

  return (json.web?.results ?? [])
    .map((item) =>
      normalizeHit(
        {
          title: item.title ?? "",
          url: item.url ?? "",
          snippet: item.description ?? "",
          source: "brave",
        },
        platform,
      ),
    )
    .filter((item): item is Giveaway => item !== null);
}
