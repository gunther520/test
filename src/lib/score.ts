import type { Giveaway } from "./types";
import { isEnterableGiveaway, qualityScore } from "./quality";

export function relevanceScore(item: Giveaway): number {
  return qualityScore(item);
}

export function isRelevant(item: Giveaway): boolean {
  return isEnterableGiveaway(item);
}
