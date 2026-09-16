import { fetchWithTimeout, ProviderError } from "../http";
import { giveawayQuery, siteQueryFor } from "../platforms";
import { normalizeHit } from "../normalize";
import type { Giveaway, PlatformFilter } from "../types";

type CseResponse = {
  items?: Array<{
    title?: string;
    link?: string;
    snippet?: string;
  }>;
  error?: { message?: string };
};

export function googleCseConfigured(): boolean {
  return Boolean(process.env.GOOGLE_API_KEY && process.env.GOOGLE_CSE_ID);
}

export async function searchGoogleCse(
  userQuery: string,
  platform: PlatformFilter,
): Promise<Giveaway[]> {
  const key = process.env.GOOGLE_API_KEY;
  const cx = process.env.GOOGLE_CSE_ID;
  if (!key || !cx) return [];

  const q = `${giveawayQuery(userQuery)} (${siteQueryFor(platform)})`;
  const params = new URLSearchParams({
    key,
    cx,
    q,
    num: "10",
    safe: "active",
  });
  const response = await fetchWithTimeout(
    `https://www.googleapis.com/customsearch/v1?${params.toString()}`,
    { headers: { accept: "application/json" } },
  );
  const json = (await response.json()) as CseResponse;
  if (!response.ok) {
    throw new ProviderError(
      json.error?.message ?? `Google CSE returned HTTP ${response.status}`,
      response.status,
    );
  }

  return (json.items ?? [])
    .map((item) =>
      normalizeHit(
        {
          title: item.title ?? "",
          url: item.link ?? "",
          snippet: item.snippet ?? "",
          source: "google-cse",
        },
        platform,
      ),
    )
    .filter((item): item is Giveaway => item !== null);
}
