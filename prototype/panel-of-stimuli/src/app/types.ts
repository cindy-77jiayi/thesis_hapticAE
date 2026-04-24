export const UI_FLOW_VALUES = ["success", "error", "notification", "loading"] as const;
export const LIKERT_KEYS = [
  "rating_match",
  "rating_meaningful",
] as const;

export type UIFlow = (typeof UI_FLOW_VALUES)[number];
export type LikertKey = (typeof LIKERT_KEYS)[number];

export type ExperimentScreen =
  | "welcome"
  | "connect"
  | "participant"
  | "flow-block"
  | "completion";

export interface FlowStimulusDefinition {
  anonymousId: number;
  displayLabel: string;
  flow: UIFlow;
  waveformSlot: number;
  gridPosition: number;
}

export interface FlowBlockResult {
  participantId: string;
  flow: UIFlow;
  anonymousId: number;
  waveformSlot: number;
  gridPosition: number;
  rating_match: number;
  rating_meaningful: number;
  timestamp: string;
}

export interface FlowOverviewResult {
  participantId: string;
  flow: UIFlow;
  top3Ids: number[];
  bottomIds: number[];
  topReason: string;
  bottomReason: string;
  timestamp: string;
}

export type FlowRatings = Partial<Record<LikertKey, number>>;

export interface LikertQuestion {
  key: LikertKey;
  prompt: string;
}
