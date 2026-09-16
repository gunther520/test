/** Leftover optional provider. Not invoked by supported free-tier search. */
import { fetchWithTimeout, ProviderError } from "../http";
import { giveawayQuery } from "../platforms";
import { normalizeHit } from "../normalize";
import type { Giveaway, PlatformFilter } from "../types";

type YoutubeSearch = {
  items?: Array<{
    id?: { videoId?: string };
    snippet?: {
      title?: string;
      description?: string;
      publishedAt?: string;
    };
  }>;
  error?: { message?: string };
};

export function youtubeConfigured(): boolean {
  return Boolean(process.env.YOUTUBE_API_KEY);
}

export async function searchYoutube(
  userQuery: string,
  platform: PlatformFilter,
): Promise<Giveaway[]> {
  if (platform !== "all" && platform !== "youtube") return [];
  const key = process.env.YOUTUBE_API_KEY;
  if (!key) return [];

  const params = new URLSearchParams({
    part: "snippet",
    type: "video",
    maxResults: "12",
    q: giveawayQuery(userQuery),
    key,
    safeSearch: "moderate",
  });
  const response = await fetchWithTimeout(
    `https://www.googleapis.com/youtube/v3/search?${params.toString()}`,
    { headers: { accept: "application/json" } },
  );
  const json = (await response.json()) as YoutubeSearch;
  if (!response.ok) {
    throw new ProviderError(
      json.error?.message ?? `YouTube API returned HTTP ${response.status}`,
      response.status,
    );
  }

  return (json.items ?? [])
    .map((item) => {
      const videoId = item.id?.videoId;
      if (!videoId) return null;
      return normalizeHit(
        {
          title: item.snippet?.title ?? "YouTube giveaway",
          url: `https://www.youtube.com/watch?v=${videoId}`,
          snippet: item.snippet?.description ?? "",
          publishedAt: item.snippet?.publishedAt,
          source: "youtube",
        },
        platform,
      );
    })
    .filter((item): item is Giveaway => item !== null);
}
