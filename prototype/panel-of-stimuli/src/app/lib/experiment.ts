import {
  BLOCK_RESULTS_CSV_HEADERS,
  FLOW_ORDER,
  OVERVIEW_CSV_HEADERS,
  STIMULI_PER_FLOW,
  TOTAL_ANON_STIMULI,
} from "../config";
import { LIKERT_KEYS } from "../types";
import type {
  FlowBlockResult,
  FlowOverviewResult,
  FlowRatings,
  FlowStimulusDefinition,
  UIFlow,
} from "../types";

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

function escapeCsvValue(value: string | number): string {
  const stringValue = String(value);
  return /[",\n]/.test(stringValue)
    ? `"${stringValue.replaceAll('"', '""')}"`
    : stringValue;
}

export function generateSeed(): string {
  if (typeof crypto !== "undefined" && "getRandomValues" in crypto) {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    return values[0].toString(36);
  }

  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
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

export function formatFlowShortLabel(uiFlow: UIFlow): string {
  const labels: Record<UIFlow, string> = {
    success: "Success",
    error: "Error",
    notification: "Notification",
    loading: "Loading",
  };

  return labels[uiFlow];
}

export function waveformSlotFromAnonymousId(anonymousId: number): number {
  return ((anonymousId - 1) % STIMULI_PER_FLOW) + 1;
}

export function buildAnonymousIdToWaveformSlot(): number[] {
  return Array.from({ length: TOTAL_ANON_STIMULI }, (_, index) => waveformSlotFromAnonymousId(index + 1));
}

export function createFlowBlocks(seed: string): Record<UIFlow, FlowStimulusDefinition[]> {
  const blocks = {} as Record<UIFlow, FlowStimulusDefinition[]>;

  FLOW_ORDER.forEach((flow, flowIndex) => {
    const blockStart = flowIndex * STIMULI_PER_FLOW + 1;
    const positions = shuffleInPlace(
      Array.from({ length: STIMULI_PER_FLOW }, (_, index) => index + 1),
      `${seed}-${flow}-grid`,
    );

    const stimuli: FlowStimulusDefinition[] = Array.from({ length: STIMULI_PER_FLOW }, (_, index) => {
      const anonymousId = blockStart + index;
      return {
        anonymousId,
        displayLabel: `A${String(anonymousId).padStart(2, "0")}`,
        flow,
        waveformSlot: waveformSlotFromAnonymousId(anonymousId),
        gridPosition: positions[index],
      };
    });

    blocks[flow] = [...stimuli].sort((left, right) => left.gridPosition - right.gridPosition);
  });

  return blocks;
}

export function isRatingsComplete(ratings: FlowRatings): boolean {
  return LIKERT_KEYS.every((key) => typeof ratings[key] === "number");
}

export function buildBlockResultsCsv(results: FlowBlockResult[]): string {
  const lines = [BLOCK_RESULTS_CSV_HEADERS.join(",")];

  const sortedResults = [...results].sort((left, right) => left.anonymousId - right.anonymousId);

  for (const result of sortedResults) {
    const row: Record<(typeof BLOCK_RESULTS_CSV_HEADERS)[number], string | number> = {
      participant_id: result.participantId,
      flow: result.flow,
      anonymous_id: result.anonymousId,
      waveform_slot: result.waveformSlot,
      grid_position: result.gridPosition,
      rating_match: result.rating_match,
      rating_appropriate: result.rating_appropriate,
      rating_meaningful: result.rating_meaningful,
      timestamp: result.timestamp,
    };

    lines.push(BLOCK_RESULTS_CSV_HEADERS.map((header) => escapeCsvValue(row[header])).join(","));
  }

  return lines.join("\n");
}

export function buildOverviewCsv(results: FlowOverviewResult[]): string {
  const lines = [OVERVIEW_CSV_HEADERS.join(",")];

  const flowIndexMap = new Map(FLOW_ORDER.map((flow, index) => [flow, index]));
  const sortedResults = [...results].sort(
    (left, right) => (flowIndexMap.get(left.flow) ?? 0) - (flowIndexMap.get(right.flow) ?? 0),
  );

  for (const result of sortedResults) {
    const row: Record<(typeof OVERVIEW_CSV_HEADERS)[number], string> = {
      participant_id: result.participantId,
      flow: result.flow,
      top3_ids: result.top3Ids.join("|"),
      bottom_ids: result.bottomIds.join("|"),
      top_reason: result.topReason,
      bottom_reason: result.bottomReason,
      timestamp: result.timestamp,
    };

    lines.push(OVERVIEW_CSV_HEADERS.map((header) => escapeCsvValue(row[header])).join(","));
  }

  return lines.join("\n");
}

export function downloadTextFile(contents: string, filename: string): void {
  const blob = new Blob([contents], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
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
