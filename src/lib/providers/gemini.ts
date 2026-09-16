import { fetchWithTimeout, ProviderError } from "../http";
import { giveawayQuery, siteQueryFor } from "../platforms";
import { normalizeHit } from "../normalize";
import type { Giveaway, PlatformFilter } from "../types";

const DEFAULT_MODEL = "gemini-2.5-flash";

type GeminiResponse = {
  candidates?: Array<{
    content?: {
      parts?: Array<{
        text?: string;
      }>;
    };
    groundingMetadata?: {
      groundingChunks?: Array<{
        web?: { uri?: string; title?: string };
      }>;
      webSearchQueries?: string[];
    };
    citationMetadata?: {
      citations?: Array<{ uri?: string; title?: string }>;
    };
  }>;
  error?: { message?: string; status?: string };
};

export function geminiConfigured(): boolean {
  return Boolean(process.env.GEMINI_API_KEY);
}

function geminiModel(): string {
  return process.env.GEMINI_MODEL?.trim() || DEFAULT_MODEL;
}

function collectUrlsFromText(text: string): Array<{ title: string; url: string }> {
  const hits: Array<{ title: string; url: string }> = [];
  const markdown = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/gi;
  let match: RegExpExecArray | null;
  while ((match = markdown.exec(text))) {
    hits.push({ title: match[1], url: match[2] });
  }
  const bare = /https?:\/\/[^\s)<>"']+/gi;
  while ((match = bare.exec(text))) {
    hits.push({ title: match[0], url: match[0] });
  }
  return hits;
}

function promptFor(userQuery: string, platform: PlatformFilter): string {
  const q = giveawayQuery(userQuery);
  const sites = siteQueryFor(platform);
  return [
    "Find currently listed public giveaway polls, lucky draws, raffles, or sweepstakes.",
    `User query: ${q}`,
    `Prefer Google Search results matching: ${sites}`,
    "Use Google Search. List real pages you found with their titles and URLs.",
    "Do not invent URLs or claim a page exists if search did not return it.",
    "Skip login walls, private accounts, and pages that are not giveaways.",
  ].join("\n");
}

export async function searchGemini(
  userQuery: string,
  platform: PlatformFilter,
): Promise<Giveaway[]> {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return [];

  const model = encodeURIComponent(geminiModel());
  const response = await fetchWithTimeout(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",
      timeoutMs: 20000,
      headers: {
        "content-type": "application/json",
        accept: "application/json",
        "x-goog-api-key": key,
      },
      body: JSON.stringify({
        contents: [{ parts: [{ text: promptFor(userQuery, platform) }] }],
        tools: [{ google_search: {} }],
      }),
    },
  );

  const json = (await response.json()) as GeminiResponse;
  if (!response.ok) {
    throw new ProviderError(
      json.error?.message ?? `Gemini returned HTTP ${response.status}`,
      response.status,
    );
  }

  const candidate = json.candidates?.[0];
  const text = candidate?.content?.parts?.map((part) => part.text ?? "").join("\n") ?? "";
  const chunks = candidate?.groundingMetadata?.groundingChunks ?? [];
  const citations = candidate?.citationMetadata?.citations ?? [];

  const raw = [
    ...chunks.map((chunk) => ({
      title: chunk.web?.title ?? "",
      url: chunk.web?.uri ?? "",
      snippet: text.slice(0, 280),
    })),
    ...citations.map((citation) => ({
      title: citation.title ?? "",
      url: citation.uri ?? "",
      snippet: text.slice(0, 280),
    })),
    ...collectUrlsFromText(text).map((hit) => ({
      title: hit.title,
      url: hit.url,
      snippet: text.slice(0, 280),
    })),
  ];

  return raw
    .map((hit) =>
      normalizeHit(
        {
          title: hit.title || "Giveaway listing",
          url: hit.url,
          snippet: hit.snippet,
          source: "gemini",
        },
        platform,
      ),
    )
    .filter((item): item is Giveaway => item !== null);
}
