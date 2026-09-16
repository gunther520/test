# Drawboard — Giveaway tracker

Web app that searches **public** social and web sources for giveaway polls, lucky draws, raffles, and similar contests.

## What it searches

| Platform | How results are found |
| --- | --- |
| Instagram, Facebook, X/Twitter, YouTube, Twitch, TikTok | Public web search with `site:` queries (DuckDuckGo HTML, Google News RSS; Brave or Google CSE if keys are set) |
| YouTube | Official YouTube Data API when `YOUTUBE_API_KEY` is set |
| Reddit and giveaway hosts (Gleam, etc.) | Reddit public JSON search + hostname classification |

The app **does not** log into social networks, solve CAPTCHAs, or bypass access controls. Indexed public URLs are sparse; optional API keys improve coverage. Some datacenters block DuckDuckGo and Reddit — Google News RSS is the unauthenticated fallback.

## Run locally

```bash
npm install
cp .env.example .env.local   # optional
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Search, filter by platform, and open a result. Empty and error states explain what to try next and how to set keys.

```bash
npm run build
npm start
```

## API keys (optional)

Copy `.env.example` to `.env.local` and restart the dev server.

| Variable | Purpose |
| --- | --- |
| `YOUTUBE_API_KEY` | YouTube Data API v3 search |
| `BRAVE_SEARCH_API_KEY` | Brave Search API (preferred web backend) |
| `GOOGLE_API_KEY` + `GOOGLE_CSE_ID` | Google Programmable Search Engine |

The UI footer lists which of these are configured.

## Adding a platform

1. Add a row to `src/lib/platforms.ts` (`id`, `hostnames`, `siteQuery`).
2. Filters, URL classification, and web-search queries pick it up automatically.
3. If the source has a documented public API, add a file under `src/lib/providers/` and register it in `src/lib/search.ts`.

## API

- `GET /api/search?q=giveaway&platform=instagram`
- `GET /api/status` — which optional keys are set
