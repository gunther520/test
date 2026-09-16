# Drawboard — Giveaway tracker

Web app that searches **public** web sources for giveaway polls, lucky draws, raffles, and similar contests. **No API keys.**

## What it searches

| Platform | How results are found |
| --- | --- |
| Instagram, Facebook, X/Twitter, YouTube, Twitch, TikTok | Public web search with `site:` queries (DuckDuckGo HTML when available, otherwise Google News RSS) |
| Reddit and giveaway hosts (Gleam, etc.) | Reddit public JSON when the host allows it, plus hostname classification |

The app **does not** log into social networks, solve CAPTCHAs, or bypass access controls. It does **not** use Gemini, Google AI Studio, Custom Search JSON API, Brave Search, or Vertex AI.

Indexed public URLs are sparse. Some datacenters block DuckDuckGo and Reddit — Google News RSS is the usual unauthenticated fallback.

## Run locally

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Search, filter by platform, and open a result. No `.env` file is required.

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
