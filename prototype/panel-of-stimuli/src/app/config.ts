import type { LikertQuestion, UIFlow } from "./types";

export const FLOW_ORDER: UIFlow[] = ["success", "error", "notification", "loading"];
export const TOTAL_FLOWS = FLOW_ORDER.length;
export const STIMULI_PER_FLOW = 15;
export const TOTAL_ANON_STIMULI = TOTAL_FLOWS * STIMULI_PER_FLOW;
export const LIKERT_VALUES = [1, 2, 3, 4, 5, 6, 7] as const;
export const LIKERT_RANGE = "1-7";
export const SERIAL_BAUD_RATE = 115200;
export const FLOW_PLAYBACK_MS = 2600;

export const BLOCK_RESULTS_CSV_HEADERS = [
  "participant_id",
  "flow",
  "anonymous_id",
  "waveform_slot",
  "grid_position",
  "rating_match",
  "rating_appropriate",
  "rating_meaningful",
  "timestamp",
] as const;

export const OVERVIEW_CSV_HEADERS = [
  "participant_id",
  "flow",
  "top3_ids",
  "bottom_ids",
  "top_reason",
  "bottom_reason",
  "timestamp",
] as const;

export const GOOGLE_FORM_CONFIG = {
  googleFormUrl: "https://forms.gle/6ZtSLwiKjy7WVybN9",
  participantPrefillEntry: "entry.139581506",
};

export const LIKERT_QUESTIONS: LikertQuestion[] = [
  { key: "rating_match", prompt: "This haptic matched the UI event." },
  { key: "rating_appropriate", prompt: "This haptic felt appropriate for this interaction." },
  { key: "rating_meaningful", prompt: "This haptic clearly conveyed a meaningful response." },
];
