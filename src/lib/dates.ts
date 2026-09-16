const MONTHS: Record<string, number> = {
  jan: 0,
  january: 0,
  feb: 1,
  february: 1,
  mar: 2,
  march: 2,
  apr: 3,
  april: 3,
  may: 4,
  jun: 5,
  june: 5,
  jul: 6,
  july: 6,
  aug: 7,
  august: 7,
  sep: 8,
  sept: 8,
  september: 8,
  oct: 9,
  october: 9,
  nov: 10,
  november: 10,
  dec: 11,
  december: 11,
};

function toIso(year: number, month: number, day: number): string | undefined {
  if (month < 0 || month > 11 || day < 1 || day > 31) return undefined;
  const date = new Date(Date.UTC(year, month, day));
  if (Number.isNaN(date.getTime())) return undefined;
  if (date.getUTCFullYear() !== year || date.getUTCMonth() !== month || date.getUTCDate() !== day) {
    return undefined;
  }
  return date.toISOString();
}

function parseNamedDate(raw: string): string | undefined {
  const dayMonthYear = raw.match(
    /\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\s+(20\d{2})\b/,
  );
  if (dayMonthYear) {
    const month = MONTHS[dayMonthYear[2].toLowerCase()];
    if (month !== undefined) {
      return toIso(Number(dayMonthYear[3]), month, Number(dayMonthYear[1]));
    }
  }
  const monthDayYear = raw.match(
    /\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b/,
  );
  if (monthDayYear) {
    const month = MONTHS[monthDayYear[1].toLowerCase()];
    if (month !== undefined) {
      return toIso(Number(monthDayYear[3]), month, Number(monthDayYear[2]));
    }
  }
  // Month + day without a year, but not "Sep 2026" (year digits after the month).
  const monthDay = raw.match(
    /\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*\d)/,
  );
  if (!monthDay) return undefined;
  const month = MONTHS[monthDay[1].toLowerCase()];
  if (month === undefined) return undefined;
  const day = Number(monthDay[2]);
  const year = new Date().getUTCFullYear();
  return toIso(year, month, day);
}

export const RECENCY_DAYS = 31;
const DAY_MS = 24 * 60 * 60 * 1000;

function parseNumericDate(text: string): string | undefined {
  const iso = text.match(/\b(20\d{2})-(\d{2})-(\d{2})\b/);
  if (iso) return toIso(Number(iso[1]), Number(iso[2]) - 1, Number(iso[3]));
  const slashed = text.match(/\/(20\d{2})\/(\d{1,2})\/(\d{1,2})(?:\/|$|\?|#)/);
  if (slashed) {
    return toIso(Number(slashed[1]), Number(slashed[2]) - 1, Number(slashed[3]));
  }
  const us = text.match(/\b(\d{1,2})\/(\d{1,2})\/(20\d{2})\b/);
  if (us) return toIso(Number(us[3]), Number(us[1]) - 1, Number(us[2]));
  return undefined;
}

function parseRelativeDate(text: string, now = Date.now()): string | undefined {
  if (/\b(?:today|hours? ago|minutes? ago|just now)\b/i.test(text)) {
    return new Date(now).toISOString();
  }
  if (/\byesterday\b/i.test(text)) {
    return new Date(now - DAY_MS).toISOString();
  }
  const rel = text.match(
    /\b(\d+)\s+(hour|hours|hr|hrs|day|days|week|weeks)\s+ago\b/i,
  );
  if (!rel) return undefined;
  const amount = Number(rel[1]);
  const unit = rel[2].toLowerCase();
  let ms = 0;
  if (unit.startsWith("hour") || unit.startsWith("hr")) ms = amount * 60 * 60 * 1000;
  else if (unit.startsWith("day")) ms = amount * DAY_MS;
  else ms = amount * 7 * DAY_MS;
  return new Date(now - ms).toISOString();
}

export function extractUrlDate(url: string): string | undefined {
  return parseNumericDate(url);
}

export function extractDates(text: string): {
  publishedAt?: string;
  endsAt?: string;
} {
  const namedEnd = text.match(
    /(?:ends?|until|closing|deadline|closes?)\s*(?:on|by|:)?\s*([^\n.]{3,40})/i,
  );

  let endsAt: string | undefined;
  if (namedEnd) {
    endsAt = parseNamedDate(namedEnd[1]) || parseNumericDate(namedEnd[1]);
  }

  const posted = text.match(
    /(?:posted|published|dated|on)\s*:?\s*([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s*20\d{2}|(20\d{2})-(\d{2})-(\d{2})|\d{1,2}\/\d{1,2}\/20\d{2}|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]{3,9}\s+20\d{2})/i,
  );
  let publishedAt = posted
    ? parseNamedDate(posted[1]) || parseNumericDate(posted[1])
    : undefined;
  if (!publishedAt) publishedAt = parseNumericDate(text);
  if (!publishedAt) publishedAt = parseNamedDate(text);
  if (!publishedAt) publishedAt = parseRelativeDate(text);
  if (publishedAt && endsAt && publishedAt === endsAt && /end|until|close|deadline/i.test(text)) {
    publishedAt = undefined;
  }

  return { publishedAt, endsAt };
}

export function isRecentGiveaway(
  item: { publishedAt?: string; endsAt?: string },
  now = Date.now(),
): boolean {
  const published = item.publishedAt ? Date.parse(item.publishedAt) : Number.NaN;
  const ends = item.endsAt ? Date.parse(item.endsAt) : Number.NaN;
  const hasPublished = !Number.isNaN(published);
  const hasEnds = !Number.isNaN(ends);
  if (!hasPublished && !hasEnds) return false;

  const cutoff = now - RECENCY_DAYS * DAY_MS;
  const skew = now + DAY_MS;
  if (hasPublished && published >= cutoff && published <= skew) return true;
  // Still-open or recently ended draws stay in the list even if the post date is older.
  if (hasEnds && ends >= cutoff) return true;
  return false;
}

export function recencyTimestamp(item: {
  publishedAt?: string;
  endsAt?: string;
}): number {
  const published = item.publishedAt ? Date.parse(item.publishedAt) : 0;
  const ends = item.endsAt ? Date.parse(item.endsAt) : 0;
  return Math.max(
    Number.isNaN(published) ? 0 : published,
    Number.isNaN(ends) ? 0 : ends,
  );
}

const MONTHS_SHORT = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

export function formatDate(iso?: string): string | undefined {
  if (!iso) return undefined;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return undefined;
  return `${date.getUTCDate()} ${MONTHS_SHORT[date.getUTCMonth()]} ${date.getUTCFullYear()}`;
}
