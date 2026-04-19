export const UI_FLOW_VALUES = ["success", "error", "notification", "loading"] as const;
export const LIKERT_KEYS = [
  "rating_match",
  "rating_pleasant",
  "rating_clarity",
  "rating_quality",
  "rating_preference",
] as const;

export type UIFlow = (typeof UI_FLOW_VALUES)[number];
export type LikertKey = (typeof LIKERT_KEYS)[number];

export type ExperimentScreen =
  | "welcome"
  | "connect"
  | "participant"
  | "trial"
  | "rating"
  | "completion";

export interface TrialDefinition {
  trialIndex: number;
  stimulusId: number;
  uiFlow: UIFlow;
}

export interface TrialResult extends TrialDefinition {
  participantId: string;
  rating_match: number;
  rating_pleasant: number;
  rating_clarity: number;
  rating_quality: number;
  rating_preference: number;
  timestamp: string;
}

export type TrialRatings = Partial<Record<LikertKey, number>>;

export interface LikertQuestion {
  key: LikertKey;
  prompt: string;
}
