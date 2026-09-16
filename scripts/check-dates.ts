import {
  extractDates,
  extractUrlDate,
  isRecentGiveaway,
  RECENCY_DAYS,
  sortGiveaways,
} from "../src/lib/dates";

const now = Date.parse("2026-09-16T12:00:00.000Z");
const day = 24 * 60 * 60 * 1000;

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const sample = extractDates(
  "Posted 15 Sep 2026. Ends Friday 18 Sep 2026",
);
assert(sample.publishedAt?.startsWith("2026-09-15") === true, `posted ${sample.publishedAt}`);
assert(sample.endsAt?.startsWith("2026-09-18") === true, `ends ${sample.endsAt}`);

const us = extractDates("Giveaway posted September 12, 2026 until 09/20/2026");
assert(us.publishedAt?.startsWith("2026-09-12") === true, `us published ${us.publishedAt}`);
assert(us.endsAt?.startsWith("2026-09-20") === true, `us ends ${us.endsAt}`);

const iso = extractDates("Published 2026-09-01. Deadline 2026-09-30");
assert(iso.publishedAt?.startsWith("2026-09-01") === true, `iso published ${iso.publishedAt}`);
assert(iso.endsAt?.startsWith("2026-09-30") === true, `iso ends ${iso.endsAt}`);

const yearTrap = extractDates("Sep 2026 giveaway with no day");
assert(!yearTrap.publishedAt, `must not parse Sep 2026 as day 20, got ${yearTrap.publishedAt}`);

const relative = extractDates("Posted 2 days ago on Instagram");
assert(Boolean(relative.publishedAt), "relative date");
assert(isRecentGiveaway(relative, now), "2 days ago is recent");

assert(
  extractUrlDate("https://blog.example.com/2026/09/10/airpods-giveaway")?.startsWith(
    "2026-09-10",
  ) === true,
  "url date",
);

assert(
  isRecentGiveaway({ publishedAt: "2026-09-01T00:00:00.000Z" }, now),
  "15 days ago should pass",
);
assert(
  !isRecentGiveaway({ publishedAt: new Date(now - 40 * day).toISOString() }, now),
  "40 days ago should drop",
);
assert(
  !isRecentGiveaway({ publishedAt: undefined, endsAt: undefined }, now),
  "undated drop",
);
assert(
  isRecentGiveaway(
    {
      publishedAt: new Date(now - 60 * day).toISOString(),
      endsAt: "2026-09-20T00:00:00.000Z",
    },
    now,
  ),
  "open draw exception",
);
assert(
  !isRecentGiveaway(
    {
      publishedAt: new Date(now - 60 * day).toISOString(),
      endsAt: new Date(now - 40 * day).toISOString(),
    },
    now,
  ),
  "old ended drop",
);

const sorted = sortGiveaways(
  [
    { publishedAt: "2026-09-10T00:00:00.000Z" },
    { publishedAt: "2026-09-01T00:00:00.000Z", endsAt: "2026-09-18T00:00:00.000Z" },
    { publishedAt: "2026-09-15T00:00:00.000Z", endsAt: "2026-09-30T00:00:00.000Z" },
  ],
  "ending",
  now,
);
assert(sorted[0].endsAt?.startsWith("2026-09-18") === true, "ending soon first");
assert(sorted[1].endsAt?.startsWith("2026-09-30") === true, "later deadline second");
assert(!sorted[2].endsAt, "no deadline last");

const newest = sortGiveaways(
  [
    { publishedAt: "2026-09-01T00:00:00.000Z" },
    { publishedAt: "2026-09-15T00:00:00.000Z" },
  ],
  "newest",
  now,
);
assert(newest[0].publishedAt?.startsWith("2026-09-15") === true, "newest first");

console.log(`ok recency=${RECENCY_DAYS}d`);
