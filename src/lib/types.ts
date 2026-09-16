export const PLATFORM_IDS = [
  "instagram",
  "facebook",
  "twitter",
  "youtube",
  "twitch",
  "reddit",
  "tiktok",
  "other",
] as const;

export type PlatformId = (typeof PLATFORM_IDS)[number];

export type PlatformFilter = PlatformId | "all";

export type ProviderId =
  | "web-search"
  | "reddit"
  | "youtube"
  | "google-cse"
  | "brave";

export type Giveaway = {
  id: string;
  title: string;
  platform: PlatformId;
  url: string;
  snippet: string;
  publishedAt?: string;
  endsAt?: string;
  source: ProviderId;
  sourceHost?: string;
};

export type ProviderStatus = {
  id: string;
  label: string;
  configured: boolean;
  used: boolean;
  ok: boolean;
  optional: boolean;
  count: number;
  error?: string;
  hint?: string;
};

export type SetupHint = {
  env: string;
  purpose: string;
  docs: string;
  set: boolean;
};

export type SearchResponse = {
  query: string;
  platform: PlatformFilter;
  results: Giveaway[];
  providers: ProviderStatus[];
  warnings: string[];
  setup: SetupHint[];
};
