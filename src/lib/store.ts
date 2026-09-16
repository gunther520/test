import { canonicalUrl } from "./normalize";
import { parsePlatformFilter } from "./platforms";
import type { Giveaway, PlatformFilter } from "./types";
import type { SortMode } from "./dates";

const STORAGE_KEY = "drawboard.tracker.v1";
const MAX_SAVED = 80;

export type SavedStatus = "active" | "entered" | "dismissed";

export type SavedGiveaway = Giveaway & {
  savedAt: string;
  status: SavedStatus;
  enteredAt?: string;
};

export type TrackerPrefs = {
  query: string;
  platform: PlatformFilter;
  sort: SortMode;
};

type StoreShape = {
  saved: SavedGiveaway[];
  prefs: TrackerPrefs;
};

const DEFAULT_PREFS: TrackerPrefs = {
  query: "giveaway",
  platform: "all",
  sort: "newest",
};

function emptyStore(): StoreShape {
  return { saved: [], prefs: { ...DEFAULT_PREFS } };
}

function isGiveawayLike(value: unknown): value is Giveaway {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<Giveaway>;
  return Boolean(item.title && item.url && item.id && item.platform && item.source);
}

function parseStore(raw: string): StoreShape {
  const parsed = JSON.parse(raw) as Partial<StoreShape>;
  const saved: SavedGiveaway[] = [];
  const seen = new Set<string>();
  for (const entry of parsed.saved ?? []) {
    if (!isGiveawayLike(entry)) continue;
    const status: SavedStatus =
      entry.status === "entered" || entry.status === "dismissed"
        ? entry.status
        : "active";
    const key = canonicalUrl(entry.url);
    if (seen.has(key)) continue;
    seen.add(key);
    saved.push({
      ...entry,
      url: entry.url,
      savedAt: entry.savedAt || new Date().toISOString(),
      status,
      enteredAt: entry.enteredAt,
    });
  }
  const prefs = parsed.prefs ?? DEFAULT_PREFS;
  const sort: SortMode = prefs.sort === "ending" ? "ending" : "newest";
  return {
    saved,
    prefs: {
      query: (prefs.query || "giveaway").trim().slice(0, 120) || "giveaway",
      platform: parsePlatformFilter(prefs.platform ?? "all"),
      sort,
    },
  };
}

function readStore(): StoreShape {
  if (typeof window === "undefined") return emptyStore();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyStore();
    return parseStore(raw);
  } catch {
    return emptyStore();
  }
}

function writeStore(next: StoreShape) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Quota or private mode — tracking still works for this session.
  }
}

export function loadPrefs(): TrackerPrefs {
  return readStore().prefs;
}

export function savePrefs(prefs: TrackerPrefs) {
  const store = readStore();
  store.prefs = {
    query: prefs.query.trim().slice(0, 120) || "giveaway",
    platform: parsePlatformFilter(prefs.platform),
    sort: prefs.sort === "ending" ? "ending" : "newest",
  };
  writeStore(store);
}

export function loadSaved(): SavedGiveaway[] {
  return readStore().saved.filter((item) => item.status !== "dismissed");
}

function upsertSaved(item: Giveaway, status: SavedStatus): SavedGiveaway[] {
  const store = readStore();
  const key = canonicalUrl(item.url);
  const existing = store.saved.find((entry) => canonicalUrl(entry.url) === key);
  const record: SavedGiveaway = {
    ...item,
    savedAt: existing?.savedAt ?? new Date().toISOString(),
    status,
    enteredAt:
      status === "entered"
        ? existing?.enteredAt ?? new Date().toISOString()
        : undefined,
  };
  store.saved = [
    record,
    ...store.saved.filter((entry) => canonicalUrl(entry.url) !== key),
  ].slice(0, MAX_SAVED);
  writeStore(store);
  return loadSaved();
}

export function saveGiveaway(item: Giveaway): SavedGiveaway[] {
  return upsertSaved(item, "active");
}

export function markEntered(url: string): SavedGiveaway[] {
  const store = readStore();
  const key = canonicalUrl(url);
  const existing = store.saved.find((entry) => canonicalUrl(entry.url) === key);
  if (!existing) return loadSaved();
  return upsertSaved(existing, "entered");
}

export function restoreActive(url: string): SavedGiveaway[] {
  const store = readStore();
  const key = canonicalUrl(url);
  const existing = store.saved.find((entry) => canonicalUrl(entry.url) === key);
  if (!existing) return loadSaved();
  return upsertSaved(existing, "active");
}

export function dismissGiveaway(url: string): SavedGiveaway[] {
  const store = readStore();
  const key = canonicalUrl(url);
  store.saved = store.saved.map((entry) =>
    canonicalUrl(entry.url) === key
      ? { ...entry, status: "dismissed" as const, enteredAt: undefined }
      : entry,
  );
  writeStore(store);
  return loadSaved();
}

export function savedStatusFor(
  saved: SavedGiveaway[],
  url: string,
): SavedStatus | undefined {
  const key = canonicalUrl(url);
  return saved.find((entry) => canonicalUrl(entry.url) === key)?.status;
}
