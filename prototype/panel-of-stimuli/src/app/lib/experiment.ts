import { CSV_HEADERS, TOTAL_STIMULI } from "../config";
import { LIKERT_KEYS, UI_FLOW_VALUES } from "../types";
import type { TrialDefinition, TrialResult, TrialRatings, UIFlow } from "../types";

function hashSeed(seed: string): number {
  let hash = 1779033703 ^ seed.length;

  for (let index = 0; index < seed.length; index += 1) {
    hash = Math.imul(hash ^ seed.charCodeAt(index), 3432918353);
    hash = (hash << 13) | (hash >>> 19);
  }

  hash = Math.imul(hash ^ (hash >>> 16), 2246822507);
  hash = Math.imul(hash ^ (hash >>> 13), 3266489909);
  return (hash ^= hash >>> 16) >>> 0;
}

function mulberry32(seed: number): () => number {
  return () => {
    let state = (seed += 0x6d2b79f5);
    state = Math.imul(state ^ (state >>> 15), state | 1);
    state ^= state + Math.imul(state ^ (state >>> 7), state | 61);
    return ((state ^ (state >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace<T>(items: T[], seed: string): T[] {
  const random = mulberry32(hashSeed(seed));
  const nextItems = [...items];

  for (let index = nextItems.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [nextItems[index], nextItems[swapIndex]] = [nextItems[swapIndex], nextItems[index]];
  }

  return nextItems;
}

export function generateSeed(): string {
  if (typeof crypto !== "undefined" && "getRandomValues" in crypto) {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    return values[0].toString(36);
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createTrialPlan(seed: string): TrialDefinition[] {
  const basePlan: Omit<TrialDefinition, "trialIndex">[] = [];

  for (const uiFlow of UI_FLOW_VALUES) {
    for (let stimulusId = 1; stimulusId <= TOTAL_STIMULI; stimulusId += 1) {
      basePlan.push({ stimulusId, uiFlow });
    }
  }

  return shuffleInPlace(basePlan, seed).map((trial, index) => ({
    ...trial,
    trialIndex: index + 1,
  }));
}

export function formatFlowLabel(uiFlow: UIFlow): string {
  const labels: Record<UIFlow, string> = {
    success: "Success payment complete",
    error: "Error failed submission",
    notification: "Notification received",
    loading: "Loading spinner complete",
  };

  return labels[uiFlow];
}

function escapeCsvValue(value: string | number): string {
  const stringValue = String(value);
  return /[",\n]/.test(stringValue)
    ? `"${stringValue.replaceAll('"', '""')}"`
    : stringValue;
}

export function buildCsv(results: TrialResult[]): string {
  const lines = [CSV_HEADERS.join(",")];

  for (const result of results) {
    const row: Record<(typeof CSV_HEADERS)[number], string | number> = {
      participant_id: result.participantId,
      trial_index: result.trialIndex,
      stimulus_id: result.stimulusId,
      ui_flow: result.uiFlow,
      rating_match: result.rating_match,
      rating_pleasant: result.rating_pleasant,
      rating_clarity: result.rating_clarity,
      rating_quality: result.rating_quality,
      rating_preference: result.rating_preference,
      timestamp: result.timestamp,
    };

    lines.push(CSV_HEADERS.map((header) => escapeCsvValue(row[header])).join(","));
  }

  return lines.join("\n");
}

export function downloadCsv(results: TrialResult[], participantId: string, seed: string): void {
  const csv = buildCsv(results);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  const safeParticipantId = participantId.replace(/[^\w-]+/g, "_");
  anchor.href = url;
  anchor.download = `${safeParticipantId || "participant"}_panel_of_stimuli_${seed}.csv`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function isRatingsComplete(ratings: TrialRatings): boolean {
  return LIKERT_KEYS.every((key) => typeof ratings[key] === "number");
}

export function buildGoogleFormUrl(
  googleFormUrl: string,
  participantPrefillEntry: string,
  participantId: string,
): string | null {
  if (!googleFormUrl.trim()) {
    return null;
  }

  const url = new URL(googleFormUrl);

  if (participantPrefillEntry.trim()) {
    url.searchParams.set(participantPrefillEntry.trim(), participantId);
    url.searchParams.set("usp", "pp_url");
  }

  return url.toString();
}
