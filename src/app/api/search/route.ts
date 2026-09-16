import { searchGiveaways } from "@/lib/search";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q") ?? "giveaway";
  const platform = searchParams.get("platform");

  try {
    const payload = await searchGiveaways(q, platform);
    const failedHard =
      payload.results.length === 0 &&
      payload.providers.filter((p) => p.used).every((p) => !p.ok);

    return Response.json(payload, { status: failedHard ? 503 : 200 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Search failed";
    return Response.json(
      {
        query: q,
        platform: platform ?? "all",
        results: [],
        providers: [],
        warnings: [message],
        setup: [],
        error: message,
      },
      { status: 500 },
    );
  }
}
