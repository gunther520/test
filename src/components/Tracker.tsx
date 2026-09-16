"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatDate, sortGiveaways, type SortMode } from "@/lib/dates";
import { FILTERABLE_PLATFORMS, getPlatform } from "@/lib/platforms";
import { displayHost, howToEnter, isUnofficialHost, shareText } from "@/lib/quality";
import {
  dismissGiveaway,
  loadPrefs,
  loadSaved,
  markEntered,
  restoreActive,
  saveGiveaway,
  savePrefs,
  savedStatusFor,
  type SavedGiveaway,
  type SavedStatus,
} from "@/lib/store";
import type {
  Giveaway,
  PlatformFilter,
  SearchResponse,
} from "@/lib/types";

const SUGGESTIONS = ["giveaway", "lucky draw", "抽獎", "Steam Deck", "AirPods", "raffle"];

type View = "search" | "saved";
type SavedPane = "active" | "entered";

function sourceLabel(source: Giveaway["source"]): string {
  switch (source) {
    case "reddit":
      return "Reddit";
    default:
      return "Web search";
  }
}

function TicketCard({
  item,
  status,
  view,
  onSave,
  onEntered,
  onDismiss,
  onRestore,
}: {
  item: Giveaway;
  status?: SavedStatus;
  view: View;
  onSave: (item: Giveaway) => void;
  onEntered: (url: string) => void;
  onDismiss: (url: string) => void;
  onRestore: (url: string) => void;
}) {
  const platform = getPlatform(item.platform);
  const published = formatDate(item.publishedAt);
  const ends = formatDate(item.endsAt);
  const entered = status === "entered";
  const saved = status === "active" || entered;
  const enterLine = howToEnter(item.title, item.snippet);
  const unofficial = isUnofficialHost(item.url, item.sourceHost);
  const [copied, setCopied] = useState(false);

  async function share() {
    const text = shareText(item.title, item.url);
    try {
      if (typeof navigator.share === "function") {
        await navigator.share({ title: item.title, text, url: item.url });
        return;
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      window.prompt("Copy this link", text);
    }
  }

  return (
    <article
      className={`ticket relative flex overflow-hidden rounded-sm border shadow-[3px_3px_0_rgba(26,18,8,0.12)] ${
        entered ? "border-ink/25 opacity-80" : "border-ink/15"
      }`}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-3 p-5 pr-6">
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
          <span className="rounded-sm bg-ink/8 px-1.5 py-0.5 text-ink">
            {platform.label}
          </span>
          <span className="normal-case tracking-normal text-[11px]">
            {displayHost(item.url, item.sourceHost)}
          </span>
          <span>{sourceLabel(item.source)}</span>
          {entered ? (
            <span className="rounded-sm bg-stamp px-1.5 py-0.5 text-ticket">
              Entered
            </span>
          ) : null}
          {status === "active" && view === "saved" ? (
            <span className="rounded-sm bg-forest px-1.5 py-0.5 text-ticket">
              Tracking
            </span>
          ) : null}
        </div>
        {ends ? (
          <p className="font-mono text-stamp">
            <span className="font-display text-[1.65rem] leading-none font-semibold">
              {ends}
            </span>
            <span className="ml-2 text-[11px] uppercase tracking-[0.16em]">
              end / draw
            </span>
          </p>
        ) : null}
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
        {enterLine ? (
          <p className="text-sm text-forest">{enterLine}</p>
        ) : null}
        {unofficial ? (
          <p className="font-mono text-[11px] text-muted">
            Unofficial host — double-check before you enter.
          </p>
        ) : null}
        <dl className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-muted">
          {published ? (
            <div className={ends ? "opacity-60" : ""}>
              <dt className="inline text-ink/45">Posted </dt>
              <dd className="inline">{published}</dd>
            </div>
          ) : null}
          {!ends ? (
            <div>
              <dt className="inline text-ink/45">Deadline </dt>
              <dd className="inline">not listed</dd>
            </div>
          ) : null}
        </dl>
        <div className="mt-auto flex flex-wrap gap-2">
          {view === "search" ? (
            <button
              type="button"
              aria-pressed={saved}
              onClick={() => {
                if (entered) return;
                if (saved) onDismiss(item.url);
                else onSave(item);
              }}
              className={`border px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] ${
                saved
                  ? "border-ink bg-ink text-ticket"
                  : "border-ink/30 bg-ticket hover:border-ink"
              }`}
            >
              {entered ? "Entered" : saved ? "Saved" : "Save"}
            </button>
          ) : entered ? (
            <>
              <button
                type="button"
                onClick={() => onRestore(item.url)}
                className="border border-ink/30 bg-ticket px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] hover:border-ink"
              >
                Still tracking
              </button>
              <button
                type="button"
                onClick={() => onDismiss(item.url)}
                className="border border-ink/30 bg-ticket px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] hover:border-ink"
              >
                Dismiss
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => onEntered(item.url)}
                className="border border-ink bg-ink px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] text-ticket hover:bg-stamp"
              >
                Mark entered
              </button>
              <button
                type="button"
                onClick={() => onDismiss(item.url)}
                className="border border-ink/30 bg-ticket px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] hover:border-ink"
              >
                Dismiss
              </button>
            </>
          )}
          <button
            type="button"
            onClick={() => void share()}
            className="border border-ink/30 bg-ticket px-2.5 py-1 font-mono text-[11px] uppercase tracking-[0.14em] hover:border-ink"
          >
            {copied ? "Copied" : "Share"}
          </button>
        </div>
      </div>
      <aside className="ticket-stub flex w-[4.6rem] shrink-0 flex-col items-center justify-between border-l border-dashed border-white/30 bg-stub px-2 py-4 text-ticket">
        <span className="font-mono text-[10px] tracking-[0.2em] uppercase">
          {ends ? "ends" : "draw"}
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

function SearchEmpty({
  query,
  platform,
}: {
  query: string;
  platform: PlatformFilter;
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
        Nothing recent enough for “{query}” on {platformLabel} in the last 31
        days. Indexed social posts are sparse, and this app does not log into
        Instagram, Facebook, or X.
      </p>
      <ul className="mx-auto mt-6 max-w-md space-y-2 text-left text-sm text-ink/80">
        <li>— Try a broader term such as giveaway or lucky draw</li>
        <li>— Switch the platform filter back to All</li>
        <li>— Only listings dated in the last 31 days are shown</li>
      </ul>
    </section>
  );
}

function SavedEmpty({ pane }: { pane: SavedPane }) {
  return (
    <section
      role="status"
      className="border border-dashed border-ink/25 bg-ticket/70 px-6 py-10 text-center"
    >
      <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted">
        {pane === "entered" ? "None entered" : "Nothing saved"}
      </p>
      <h2 className="mt-3 font-display text-3xl">
        {pane === "entered" ? "No entered draws yet" : "No saved giveaways"}
      </h2>
      <p className="mx-auto mt-3 max-w-lg text-ink/75">
        {pane === "entered"
          ? "Mark a saved ticket as entered when you have joined the draw. It leaves the active list and stays here so you can reopen the post."
          : "Save a ticket from search to track it on this device. Saved draws stay in your browser — nothing is sent to a server."}
      </p>
    </section>
  );
}

function ErrorState({
  message,
  warnings,
  onRetry,
}: {
  message: string;
  warnings: string[];
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
        Drawboard uses free public web search only (DuckDuckGo or Google News
        RSS, plus Reddit when allowed). It does not scrape login walls,
        CAPTCHAs, or private accounts, and it does not use paid search APIs.
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-6 border border-ink bg-ink px-4 py-2 font-mono text-xs uppercase tracking-[0.18em] text-ticket hover:bg-stamp"
      >
        Retry search
      </button>
    </section>
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
  const [sort, setSort] = useState<SortMode>("newest");
  const [view, setView] = useState<View>("search");
  const [savedPane, setSavedPane] = useState<SavedPane>("active");
  const [saved, setSaved] = useState<SavedGiveaway[]>([]);
  const [data, setData] = useState<SearchResponse>(initial);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(initialError);
  const searchSeq = useRef(0);
  const sortRef = useRef<SortMode>("newest");
  const restored = useRef(false);

  const persist = useCallback(
    (nextQuery: string, nextPlatform: PlatformFilter, nextSort: SortMode) => {
      savePrefs({ query: nextQuery, platform: nextPlatform, sort: nextSort });
    },
    [],
  );

  const runSearch = useCallback(
    async (nextQuery: string, nextPlatform: PlatformFilter) => {
      const q = nextQuery.trim() || "giveaway";
      const seq = ++searchSeq.current;
      setQuery(q);
      setDraft(q);
      setPlatform(nextPlatform);
      setView("search");
      setLoading(true);
      setError(null);
      persist(q, nextPlatform, sortRef.current);
      try {
        const params = new URLSearchParams({
          q,
          platform: nextPlatform,
        });
        const response = await fetch(`/api/search?${params.toString()}`);
        const payload = (await response.json()) as SearchResponse & {
          error?: string;
        };
        if (seq !== searchSeq.current) return;
        setData(payload);
        if (!response.ok && payload.results.length === 0) {
          setError(
            payload.error ?? payload.warnings[0] ?? "Search backends failed",
          );
        }
      } catch {
        if (seq !== searchSeq.current) return;
        setError(
          "The app could not reach its own search API. Is the dev server running?",
        );
      } finally {
        if (seq === searchSeq.current) setLoading(false);
      }
    },
    [persist],
  );

  useEffect(() => {
    if (restored.current) return;
    restored.current = true;
    const prefs = loadPrefs();
    sortRef.current = prefs.sort;
    const timer = window.setTimeout(() => {
      setSort(prefs.sort);
      setSaved(loadSaved());
      if (prefs.query !== initial.query || prefs.platform !== initial.platform) {
        void runSearch(prefs.query, prefs.platform);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [initial.platform, initial.query, runSearch]);

  const visibleSearch = useMemo(() => {
    const results = data?.results ?? [];
    const filtered =
      platform === "all"
        ? results
        : results.filter((item) => item.platform === platform);
    return sortGiveaways(filtered, sort);
  }, [data, platform, sort]);

  const activeSaved = useMemo(
    () =>
      sortGiveaways(
        saved.filter((item) => item.status === "active"),
        sort,
      ),
    [saved, sort],
  );
  const enteredSaved = useMemo(
    () =>
      sortGiveaways(
        saved.filter((item) => item.status === "entered"),
        sort,
      ),
    [saved, sort],
  );
  const savedVisible = savedPane === "entered" ? enteredSaved : activeSaved;

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const next = draft.trim() || "giveaway";
    setDraft(next);
    void runSearch(next, platform);
  }

  function changeSort(next: SortMode) {
    sortRef.current = next;
    setSort(next);
    persist(query, platform, next);
  }

  const hardError = Boolean(
    view === "search" && error && visibleSearch.length === 0 && !loading,
  );

  const cardHandlers = {
    onSave: (item: Giveaway) => setSaved(saveGiveaway(item)),
    onEntered: (url: string) => {
      setSaved(markEntered(url));
      setSavedPane("entered");
    },
    onDismiss: (url: string) => setSaved(dismissGiveaway(url)),
    onRestore: (url: string) => {
      setSaved(restoreActive(url));
      setSavedPane("active");
    },
  };

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 py-8 sm:px-6">
      <header className="border-b-2 border-ink pb-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.28em] text-stamp">
              Hoi Kan NG · public tracker
            </p>
            <h1 className="mt-2 font-display text-5xl tracking-tight sm:text-6xl">
              Drawboard
            </h1>
            <p className="mt-2 max-w-xl text-lg text-ink/75">
              Find public giveaways, save the ones you mean to enter, and mark
              them entered when you have joined.
            </p>
          </div>
          <p className="max-w-xs font-mono text-[11px] leading-5 text-muted">
            Saves stay in this browser. Search is free public web + Reddit JSON
            when that host allows it — last 31 days only.
          </p>
        </div>

        <div
          className="mt-6 flex flex-wrap gap-2"
          role="tablist"
          aria-label="Search or saved"
        >
          <FilterChip
            active={view === "search"}
            onClick={() => setView("search")}
            label="Search"
          />
          <FilterChip
            active={view === "saved"}
            onClick={() => setView("saved")}
            label={`Saved ${activeSaved.length}`}
          />
        </div>

        <form onSubmit={submit} className="mt-6 flex flex-col gap-3 sm:flex-row">
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
            onClick={() => void runSearch(draft, "all")}
            label="All"
          />
          {FILTERABLE_PLATFORMS.map((item) => (
            <FilterChip
              key={item.id}
              active={platform === item.id}
              onClick={() => void runSearch(draft, item.id)}
              label={item.label}
            />
          ))}
        </div>
      </header>

      <div className="mt-5 flex flex-wrap items-center justify-between gap-3 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
        <p>
          {view === "saved"
            ? `${savedVisible.length} ${savedPane === "entered" ? "entered" : "tracking"}`
            : loading
              ? "Searching public sources…"
              : `${visibleSearch.length} result${visibleSearch.length === 1 ? "" : "s"} · last 31 days · ${query}`}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          {view === "saved" ? (
            <>
              <FilterChip
                active={savedPane === "active"}
                onClick={() => setSavedPane("active")}
                label={`Tracking ${activeSaved.length}`}
              />
              <FilterChip
                active={savedPane === "entered"}
                onClick={() => setSavedPane("entered")}
                label={`Entered ${enteredSaved.length}`}
              />
            </>
          ) : data ? (
            <p>
              {data.providers
                .filter((provider) => provider.used && provider.ok)
                .map((provider) => provider.label)
                .join(" · ") || "no live backends"}
            </p>
          ) : null}
        </div>
      </div>

      <div
        className="mt-3 flex flex-wrap gap-2"
        role="group"
        aria-label="Sort results"
      >
        <FilterChip
          active={sort === "newest"}
          onClick={() => changeSort("newest")}
          label="Newest"
        />
        <FilterChip
          active={sort === "ending"}
          onClick={() => changeSort("ending")}
          label="Ending soon"
        />
      </div>

      {view === "search" && data?.warnings.length && !hardError ? (
        <p className="mt-3 border border-ink/15 bg-ticket px-3 py-2 text-sm text-ink/70">
          {data.warnings.join(" · ")}
        </p>
      ) : null}

      <main className="mt-6 flex-1">
        {view === "saved" ? (
          savedVisible.length === 0 ? (
            <SavedEmpty pane={savedPane} />
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              {savedVisible.map((item) => (
                <TicketCard
                  key={item.url}
                  item={item}
                  status={item.status}
                  view="saved"
                  {...cardHandlers}
                />
              ))}
            </div>
          )
        ) : loading ? (
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
            onRetry={() => void runSearch(query, platform)}
          />
        ) : visibleSearch.length === 0 ? (
          <SearchEmpty query={query} platform={platform} />
        ) : (
          <div className="grid gap-4 md:grid-cols-2">
            {visibleSearch.map((item) => (
              <TicketCard
                key={item.id + item.url}
                item={item}
                status={savedStatusFor(saved, item.url)}
                view="search"
                {...cardHandlers}
              />
            ))}
          </div>
        )}
      </main>

      <footer className="mt-12 border-t border-ink/20 py-6 text-sm text-muted">
        <details>
          <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.18em]">
            How this tracker works
          </summary>
          <p className="mt-4 max-w-2xl">
            Drawboard searches public web indexes with DuckDuckGo HTML when
            available, otherwise Google News RSS using each platform’s{" "}
            <code className="font-mono text-xs">site:</code> pattern. Reddit’s
            public JSON API is used when that host allows it. Only listings
            dated in the last 31 days are shown; undated hits are dropped. The
            list keeps social posts and contest hosts, not news roundups; a lone
            “giveaway” is not enough. Save, entered, and the last search stay in
            localStorage on this device. No API keys, no Gemini, no Custom
            Search, no Brave, and no Vertex.
          </p>
          <p className="mt-3 max-w-2xl">
            To track another public source, add a row in{" "}
            <code className="font-mono text-xs">src/lib/platforms.ts</code>{" "}
            (id, hostnames, site query).
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
