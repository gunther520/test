# Drawboard — Giveaway tracker

Web app that searches **public** web sources for giveaway polls, lucky draws, raffles, and similar contests, then lets you **save** tickets locally, mark them entered, and sort by newest or ending soon. **No API keys.**

## What it searches

| Platform | How results are found |
| --- | --- |
| Instagram, Facebook, X/Twitter, YouTube, Twitch, TikTok | Public web search with `site:` queries (DuckDuckGo HTML when available, otherwise Google News RSS) |
| Reddit and giveaway hosts (Gleam, etc.) | Reddit public JSON when the host allows it, plus hostname classification |

The app **does not** log into social networks, solve CAPTCHAs, or bypass access controls. It does **not** use Gemini, Google AI Studio, Custom Search JSON API, Brave Search, or Vertex AI.

Indexed public URLs are sparse. Some datacenters block DuckDuckGo and Reddit — Google News RSS is the usual unauthenticated fallback.

Only listings dated in the **last 31 days** are shown (RSS `pubDate`, Reddit `created_utc`, or a date parsed from the title/snippet). Hits with no usable date are dropped so old undated posts do not appear.

The list prefers **enterable social posts and contest hosts** (Instagram `/p/` `/reel/`, Facebook posts, X `/status/`, YouTube watch/shorts, TikTok `/video/`, Reddit comments, Gleam/Woobox). Generic news/blog hosts and how-to roundups are dropped. A lone word “giveaway” is not enough unless the URL is a social post. Each card shows the source host.

## Tracker (this device)

- **Save** a result to a local Saved list (browser `localStorage`, keyed by URL).
- **Mark entered** moves it off the active list (still visible under Entered). **Dismiss** removes it from Saved.
- **Sort** Search and Saved by **Newest** (posted/parsed date) or **Ending soon** (listed deadlines first).
- The last search query, platform filter, and sort are remembered on this browser.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Search, filter by platform, save a ticket, and open a result. No `.env` file is required.

```bash
npm run build
npm start
```

## Adding a platform

1. Add a row to `src/lib/platforms.ts` (`id`, `hostnames`, `siteQuery`).
2. Filters, URL classification, and public `site:` queries pick it up automatically.

## API

- `GET /api/search?q=giveaway&platform=instagram`
- `GET /api/status` — confirms free public search (no keys)
