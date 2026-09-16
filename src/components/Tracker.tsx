"use client";

import { useCallback, useMemo, useState } from "react";
import { formatDate } from "@/lib/dates";
import { FILTERABLE_PLATFORMS, getPlatform } from "@/lib/platforms";
import type {
  Giveaway,
  PlatformFilter,
  SearchResponse,
  SetupHint,
} from "@/lib/types";

const SUGGESTIONS = ["giveaway", "lucky draw", "Steam Deck", "AirPods", "raffle"];

function sourceLabel(source: Giveaway["source"]): string {
  switch (source) {
    case "youtube":
      return "YouTube API";
    case "reddit":
      return "Reddit";
    case "brave":
      return "Brave";
    case "google-cse":
      return "Google CSE";
    default:
      return "Web search";
  }
}

function TicketCard({ item }: { item: Giveaway }) {
  const platform = getPlatform(item.platform);
  const published = formatDate(item.publishedAt);
  const ends = formatDate(item.endsAt);

  return (
    <article className="ticket relative flex overflow-hidden rounded-sm border border-ink/15 shadow-[3px_3px_0_rgba(26,18,8,0.12)]">
      <div className="flex min-w-0 flex-1 flex-col gap-3 p-5 pr-6">
        <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
          <span className="rounded-sm bg-ink/8 px-1.5 py-0.5 text-ink">
            {platform.label}
          </span>
          <span>{sourceLabel(item.source)}</span>
        </div>
        <h2 className="font-display text-[1.35rem] leading-snug font-semibold text-pretty">
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="decoration-stamp/40 underline-offset-4 hover:underline"
          >
            {item.title}
          </a>
        </h2>
        {item.snippet ? (
          <p className="text-[15px] leading-6 text-ink/75">{item.snippet}</p>
        ) : null}
        <dl className="mt-auto flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
          {published ? (
            <div>
              <dt className="inline text-ink/45">Posted </dt>
              <dd className="inline">{published}</dd>
            </div>
          ) : null}
          {ends ? (
            <div>
              <dt className="inline text-stamp">Ends </dt>
              <dd className="inline text-stamp">{ends}</dd>
            </div>
          ) : (
            <div>
              <dt className="inline text-ink/45">Dates </dt>
              <dd className="inline">not listed</dd>
            </div>
          )}
        </dl>
      </div>
      <aside className="ticket-stub flex w-[4.6rem] shrink-0 flex-col items-center justify-between border-l border-dashed border-white/30 bg-stub px-2 py-4 text-ticket">
        <span className="font-mono text-[10px] tracking-[0.2em] uppercase">
          draw
        </span>
        <span
          className="font-display text-2xl font-semibold"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          {platform.stub}
        </span>
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="font-mono text-[10px] tracking-widest uppercase underline underline-offset-2"
        >
          Open
        </a>
      </aside>
    </article>
  );
}

function EmptyState({
  query,
  platform,
  setup,
}: {
  query: string;
  platform: PlatformFilter;
  setup: SetupHint[];
}) {
  const platformLabel =
    platform === "all" ? "all platforms" : getPlatform(platform).label;
  return (
    <section
      role="status"
      className="border border-dashed border-ink/25 bg-ticket/70 px-6 py-10 text-center"
    >
      <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">
        No tickets
      </p>
      <h2 className="mt-3 font-display text-3xl">No public giveaways matched</h2>
      <p className="mx-auto mt-3 max-w-lg text-ink/75">
        Nothing turned up for “{query}” on {platformLabel}. Indexed social posts
        are sparse, and this app does not log into Instagram, Facebook, or X.
      </p>
      <ul className="mx-auto mt-6 max-w-md space-y-2 text-left text-sm text-ink/80">
        <li>— Try a broader term such as giveaway or lucky draw</li>
        <li>— Switch the platform filter back to All</li>
        <li>— Add an optional API key for deeper official search</li>
      </ul>
      <SetupList setup={setup} compact />
    </section>
  );
}

function ErrorState({
  message,
  warnings,
  setup,
  onRetry,
}: {
  message: string;
  warnings: string[];
  setup: SetupHint[];
  onRetry: () => void;
}) {
  return (
    <section
      role="alert"
      className="border border-stamp/40 bg-[#fbe8e2] px-6 py-10"
    >
      <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-stamp">
        Search failed
      </p>
      <h2 className="mt-3 font-display text-3xl">Couldn’t reach search sources</h2>
      <p className="mt-3 max-w-2xl text-ink/80">{message}</p>
      {warnings.length > 0 ? (
        <ul className="mt-4 space-y-1 font-mono text-xs text-ink/70">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      ) : null}
      <p className="mt-4 text-sm text-ink/70">
        Drawboard uses public web search and optional official APIs. It does not
        scrape login walls, CAPTCHAs, or private accounts.
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-6 border border-ink bg-ink px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-ticket hover:bg-stamp"
      >
        Retry search
      </button>
      <SetupList setup={setup} />
    </section>
  );
}

function SetupList({
  setup,
  compact = false,
}: {
  setup: SetupHint[];
  compact?: boolean;
}) {
  if (setup.length === 0) return null;
  return (
    <div className={`text-left ${compact ? "mx-auto mt-8 max-w-xl" : "mt-8"}`}>
      <h3 className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
        How to set API keys
      </h3>
      <p className="mt-2 text-sm text-ink/70">
        Copy <code className="font-mono text-xs">.env.example</code> to{" "}
        <code className="font-mono text-xs">.env.local</code>, fill any keys you
        have, then restart <code className="font-mono text-xs">npm run dev</code>.
      </p>
      <ul className="mt-3 divide-y divide-ink/10 border border-ink/10 bg-ticket">
        {setup.map((hint) => (
          <li key={hint.env} className="flex flex-col gap-1 px-4 py-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="font-mono text-sm">{hint.env}</p>
              <p className="text-sm text-ink/70">{hint.purpose}</p>
            </div>
            <span
              className={`mt-1 font-mono text-[10px] uppercase tracking-widest ${
                hint.set ? "text-forest" : "text-muted"
              }`}
            >
              {hint.set ? "configured" : "optional"}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Tracker({
  initial,
  initialError,
}: {
  initial: SearchResponse;
  initialError: string | null;
}) {
  const [query, setQuery] = useState(initial.query);
  const [draft, setDraft] = useState(initial.query);
  const [platform, setPlatform] = useState<PlatformFilter>(initial.platform);
  const [data, setData] = useState<SearchResponse>(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(initialError);

  const runSearch = useCallback(async (nextQuery: string, nextPlatform: PlatformFilter) => {
    setQuery(nextQuery);
    setPlatform(nextPlatform);
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        q: nextQuery,
        platform: nextPlatform,
      });
      const response = await fetch(`/api/search?${params.toString()}`);
      const payload = (await response.json()) as SearchResponse & { error?: string };
      setData(payload);
      if (!response.ok && payload.results.length === 0) {
        setError(payload.error ?? payload.warnings[0] ?? "Search backends failed");
      }
    } catch {
      setError("The app could not reach its own search API. Is the dev server running?");
    } finally {
      setLoading(false);
    }
  }, []);

  const visible = useMemo(() => {
    const results = data?.results ?? [];
    if (platform === "all") return results;
    return results.filter((item) => item.platform === platform);
  }, [data, platform]);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const next = draft.trim() || "giveaway";
    setDraft(next);
    void runSearch(next, platform);
  }

  const hardError = Boolean(error && visible.length === 0 && !loading);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 py-8 sm:px-6">
      <header className="border-b-2 border-ink pb-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-stamp">
              Hoi Kan NG · public search
            </p>
            <h1 className="mt-2 font-display text-5xl tracking-tight sm:text-6xl">
              Drawboard
            </h1>
            <p className="mt-2 max-w-xl text-lg text-ink/75">
              Find public giveaway polls and lucky draws across Instagram,
              Facebook, X, YouTube, Twitch, and similar sources.
            </p>
          </div>
          <p className="max-w-xs font-mono text-[11px] leading-5 text-muted">
            No login scraping. Results come from public web search, Reddit JSON,
            and official APIs when keys are set.
          </p>
        </div>

        <form onSubmit={submit} className="mt-8 flex flex-col gap-3 sm:flex-row">
          <label className="sr-only" htmlFor="giveaway-query">
            Search giveaways
          </label>
          <input
            id="giveaway-query"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Search giveaways, prizes, or brands"
            className="flex-1 border border-ink bg-ticket px-4 py-3 font-display text-xl outline-none ring-stamp/30 focus:ring-2"
          />
          <button
            type="submit"
            className="border border-ink bg-stamp px-6 py-3 font-mono text-xs uppercase tracking-[0.2em] text-ticket hover:bg-ink"
          >
            Find draws
          </button>
        </form>

        <div className="mt-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => {
                setDraft(suggestion);
                void runSearch(suggestion, platform);
              }}
              className="border border-ink/20 bg-ticket px-2 py-1 font-mono text-[11px] uppercase tracking-wide hover:border-ink"
            >
              {suggestion}
            </button>
          ))}
        </div>

        <div
          className="mt-5 flex flex-wrap gap-2"
          role="group"
          aria-label="Filter by platform"
        >
          <FilterChip
            active={platform === "all"}
            onClick={() => void runSearch(query, "all")}
            label="All"
          />
          {FILTERABLE_PLATFORMS.map((item) => (
            <FilterChip
              key={item.id}
              active={platform === item.id}
              onClick={() => void runSearch(query, item.id)}
              label={item.label}
            />
          ))}
        </div>
      </header>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
        <p>
          {loading
            ? "Searching public sources…"
            : `${visible.length} result${visible.length === 1 ? "" : "s"} · ${query}`}
        </p>
        {data ? (
          <p>
            {data.providers
              .filter((provider) => provider.used && provider.ok)
              .map((provider) => provider.label)
              .join(" · ") || "no live backends"}
          </p>
        ) : null}
      </div>

      {data?.warnings.length && !hardError ? (
        <p className="mt-3 border border-ink/15 bg-ticket px-3 py-2 text-sm text-ink/70">
          {data.warnings.join(" · ")}
        </p>
      ) : null}

      <main className="mt-6 flex-1">
        {loading ? (
          <div className="grid gap-4 md:grid-cols-2" aria-busy="true">
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className="h-44 animate-pulse border border-ink/10 bg-ticket/80"
              />
            ))}
          </div>
        ) : hardError ? (
          <ErrorState
            message={error ?? "Search failed"}
            warnings={data?.warnings ?? []}
            setup={data?.setup ?? []}
            onRetry={() => void runSearch(query, platform)}
          />
        ) : visible.length === 0 ? (
          <EmptyState
            query={query}
            platform={platform}
            setup={data?.setup ?? []}
          />
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {visible.map((item) => (
              <TicketCard key={item.id + item.url} item={item} />
            ))}
          </div>
        )}
      </main>

      <footer className="mt-12 border-t border-ink/20 py-6 text-sm text-muted">
        <details>
          <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.18em]">
            API keys & adding a platform
          </summary>
          <SetupList setup={data?.setup ?? []} />
          <p className="mt-4 max-w-2xl">
            To track another public source, add a row in{" "}
            <code className="font-mono text-xs">src/lib/platforms.ts</code>{" "}
            (id, hostnames, site query). Optional official APIs live under{" "}
            <code className="font-mono text-xs">src/lib/providers/</code>.
          </p>
        </details>
      </footer>
    </div>
  );
}

function FilterChip({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.14em] ${
        active
          ? "border-ink bg-ink text-ticket"
          : "border-ink/25 bg-ticket hover:border-ink"
      }`}
    >
      {label}
    </button>
  );
}
