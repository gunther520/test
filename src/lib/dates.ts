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
  return date.toISOString();
}

function parseNamedDate(raw: string): string | undefined {
  const match = raw.match(
    /([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(20\d{2}))?/,
  );
  if (!match) return undefined;
  const month = MONTHS[match[1].toLowerCase()];
  if (month === undefined) return undefined;
  const day = Number(match[2]);
  const year = match[3] ? Number(match[3]) : new Date().getUTCFullYear();
  return toIso(year, month, day);
}

export function extractDates(text: string): {
  publishedAt?: string;
  endsAt?: string;
} {
  const iso = text.match(/\b(20\d{2})-(\d{2})-(\d{2})\b/);
  const us = text.match(/\b(\d{1,2})\/(\d{1,2})\/(20\d{2})\b/);
  const named = text.match(
    /(?:ends?|until|closing|deadline|draw(?:s|ing)?(?:\s+on)?|closes?)\s*(?:on|by|:)?\s*([A-Za-z]{3,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s*(?:20\d{2})?)/i,
  );

  let endsAt: string | undefined;
  if (named) endsAt = parseNamedDate(named[1]);
  if (!endsAt && iso && /end|until|close|deadline|draw/i.test(text)) {
    endsAt = toIso(Number(iso[1]), Number(iso[2]) - 1, Number(iso[3]));
  }
  if (!endsAt && us && /end|until|close|deadline|draw/i.test(text)) {
    endsAt = toIso(Number(us[3]), Number(us[1]) - 1, Number(us[2]));
  }

  return { endsAt };
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
