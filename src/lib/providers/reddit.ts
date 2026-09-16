import { decodeHtml, fetchWithTimeout, ProviderError } from "../http";
import { giveawayQuery } from "../platforms";
import { normalizeHit } from "../normalize";
import type { Giveaway, PlatformFilter } from "../types";

type RedditListing = {
  data?: {
    children?: Array<{
      data?: {
        title?: string;
        url?: string;
        permalink?: string;
        selftext?: string;
        created_utc?: number;
      };
    }>;
  };
};

export async function searchReddit(
  userQuery: string,
  platform: PlatformFilter,
): Promise<Giveaway[]> {
  if (platform !== "all" && platform !== "reddit" && platform !== "other") {
    return [];
  }

  const q = giveawayQuery(userQuery);
  const params = new URLSearchParams({
    q,
    sort: "new",
    t: "month",
    limit: "25",
    raw_json: "1",
  });
  const url = `https://www.reddit.com/search.json?${params.toString()}`;
  const response = await fetchWithTimeout(url, {
    headers: { accept: "application/json" },
  });
  if (!response.ok) {
    throw new ProviderError(`Reddit returned HTTP ${response.status}`, response.status);
  }

  const json = (await response.json()) as RedditListing;
  const children = json.data?.children ?? [];
  return children
    .map((child) => {
      const post = child.data;
      if (!post?.title) return null;
      const postUrl = post.url?.startsWith("http")
        ? post.url
        : `https://www.reddit.com${post.permalink ?? ""}`;
      return normalizeHit(
        {
          title: decodeHtml(post.title),
          url: postUrl,
          snippet: decodeHtml(post.selftext ?? "").slice(0, 280),
          publishedAt: post.created_utc
            ? new Date(post.created_utc * 1000).toISOString()
            : undefined,
          source: "reddit",
        },
        platform === "other" ? "all" : platform,
      );
    })
    .filter((item): item is Giveaway => item !== null);
}
