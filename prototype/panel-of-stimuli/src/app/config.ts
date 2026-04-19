import type { LikertQuestion } from "./types";

export const TOTAL_STIMULI = 15;
export const TOTAL_TRIALS = 60;
export const LIKERT_VALUES = [1, 2, 3, 4, 5, 6, 7] as const;
export const SERIAL_BAUD_RATE = 115200;
export const FLOW_PLAYBACK_MS = 2600;

export const CSV_HEADERS = [
  "participant_id",
  "trial_index",
  "stimulus_id",
  "ui_flow",
  "rating_match",
  "rating_pleasant",
  "rating_clarity",
  "rating_quality",
  "rating_preference",
  "timestamp",
] as const;

export const GOOGLE_FORM_CONFIG = {
  googleFormUrl: "",
  participantPrefillEntry: "",
};

export const LIKERT_QUESTIONS: LikertQuestion[] = [
  { key: "rating_match", prompt: "This haptic matched the UI event." },
  { key: "rating_pleasant", prompt: "This haptic felt pleasant." },
  { key: "rating_clarity", prompt: "This haptic clearly communicated the event." },
  { key: "rating_quality", prompt: "This haptic felt polished / high quality." },
  { key: "rating_preference", prompt: "Overall preference." },
];
