import type { PlatformFilter, PlatformId } from "./types";

export type PlatformDef = {
  id: PlatformId;
  label: string;
  short: string;
  /** Hostnames used to classify a result URL. */
  hostnames: string[];
  /** DuckDuckGo / Google / Brave site: operator. */
  siteQuery: string;
  /** Words that hint this platform when the URL is a news aggregator. */
  keywords: string[];
  stub: string;
};

/**
 * Add a platform here and it shows up in filters, URL classification,
 * and public web-search site: queries. No scraper required.
 */
export const PLATFORMS: PlatformDef[] = [
  {
    id: "instagram",
    label: "Instagram",
    short: "IG",
    hostnames: ["instagram.com", "instagr.am"],
    siteQuery: "site:instagram.com",
    keywords: ["instagram", "insta"],
    stub: "IG",
  },
  {
    id: "facebook",
    label: "Facebook",
    short: "FB",
    hostnames: ["facebook.com", "fb.com", "fb.watch"],
    siteQuery: "site:facebook.com",
    keywords: ["facebook"],
    stub: "FB",
  },
  {
    id: "twitter",
    label: "X / Twitter",
    short: "X",
    hostnames: ["twitter.com", "x.com", "mobile.twitter.com"],
    siteQuery: "(site:x.com OR site:twitter.com)",
    keywords: ["twitter", "tweet", " on x "],
    stub: "X",
  },
  {
    id: "youtube",
    label: "YouTube",
    short: "YT",
    hostnames: ["youtube.com", "youtu.be", "m.youtube.com"],
    siteQuery: "site:youtube.com",
    keywords: ["youtube"],
    stub: "YT",
  },
  {
    id: "twitch",
    label: "Twitch",
    short: "TW",
    hostnames: ["twitch.tv"],
    siteQuery: "site:twitch.tv",
    keywords: ["twitch"],
    stub: "TW",
  },
  {
    id: "reddit",
    label: "Reddit",
    short: "RD",
    hostnames: ["reddit.com", "old.reddit.com"],
    siteQuery: "site:reddit.com",
    keywords: ["reddit"],
    stub: "RD",
  },
  {
    id: "tiktok",
    label: "TikTok",
    short: "TT",
    hostnames: ["tiktok.com", "vm.tiktok.com"],
    siteQuery: "site:tiktok.com",
    keywords: ["tiktok"],
    stub: "TT",
  },
  {
    id: "other",
    label: "Other",
    short: "••",
    hostnames: ["gleam.io", "woobox.com", "kingsumo.com", "rafflecopter.com"],
    siteQuery: "(site:gleam.io OR site:woobox.com OR site:kingsumo.com)",
    keywords: ["gleam", "rafflecopter", "lucky draw"],
    stub: "OT",
  },
];

export const FILTERABLE_PLATFORMS: PlatformDef[] = PLATFORMS.filter(
  (p) => p.id !== "other",
);

const HOST_INDEX = new Map<string, PlatformId>();
for (const platform of PLATFORMS) {
  for (const host of platform.hostnames) {
    HOST_INDEX.set(host, platform.id);
  }
}

export function getPlatform(id: PlatformId): PlatformDef {
  return PLATFORMS.find((p) => p.id === id) ?? PLATFORMS[PLATFORMS.length - 1];
}

export function parsePlatformFilter(value: string | null): PlatformFilter {
  if (!value || value === "all") return "all";
  return PLATFORMS.some((p) => p.id === value) ? (value as PlatformId) : "all";
}

export function classifyPlatform(url: string, title = ""): PlatformId {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "").toLowerCase();
    const exact = HOST_INDEX.get(host);
    if (exact) return exact;
    for (const [known, id] of HOST_INDEX) {
      if (host === known || host.endsWith(`.${known}`)) return id;
    }
  } catch {
    // ignore invalid URLs
  }

  const haystack = ` ${title.toLowerCase()} `;
  for (const platform of PLATFORMS) {
    if (platform.id === "other") continue;
    if (platform.keywords.some((kw) => haystack.includes(kw))) {
      return platform.id;
    }
  }
  return "other";
}

export function siteQueryFor(filter: PlatformFilter): string {
  if (filter === "all") {
    return PLATFORMS.filter((p) =>
      ["instagram", "facebook", "twitter", "youtube", "twitch", "tiktok"].includes(
        p.id,
      ),
    )
      .map((p) => p.siteQuery)
      .join(" OR ");
  }
  return getPlatform(filter).siteQuery;
}

export function giveawayQuery(userQuery: string): string {
  const trimmed = userQuery.trim() || "giveaway";
  const alreadyHasTerm =
    /giveaway|lucky draw|raffle|sweepstake|contest|抽獎|免費送/i.test(trimmed);
  if (alreadyHasTerm) return trimmed;
  return `${trimmed} (giveaway OR "lucky draw" OR raffle OR sweepstakes OR "comment to win" OR 抽獎)`;
}
