import type { Giveaway } from "./types";

export function relevanceScore(item: Giveaway): number {
  const text = `${item.title} ${item.snippet}`.toLowerCase();
  let score = 0;
  if (/lucky draw/.test(text)) score += 5;
  if (/giveaway poll|poll to win/.test(text)) score += 5;
  if (/giveaway/.test(text)) score += 3;
  if (/raffle|sweepstake/.test(text)) score += 3;
  if (/comment to win|retweet to win|rt to win|tag a friend|follow to enter|enter to win/.test(text)) {
    score += 4;
  }
  if (/\bwin a\b|\bwin an\b/.test(text)) score += 2;
  if (/contest/.test(text)) score += 1;
  if (item.endsAt) score += 2;
  if (/giveaway to (israel|ukraine|china)|stimulus giveaway|tax giveaway/.test(text)) {
    score -= 5;
  }
  return score;
}

export function isRelevant(item: Giveaway): boolean {
  return relevanceScore(item) >= 3;
}
