import type { ReactNode } from "react";
import { startTransition, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Cable,
  ChevronRight,
  Download,
  ExternalLink,
  Maximize,
  Minimize,
  Play,
  RefreshCcw,
  RotateCcw,
  ShieldEllipsis,
  Sparkles,
  Usb,
} from "lucide-react";

import { FLOW_PLAYBACK_MS, GOOGLE_FORM_CONFIG, LIKERT_QUESTIONS, LIKERT_VALUES, TOTAL_TRIALS } from "./config";
import { FlowRenderer } from "./components/FlowRenderer";
import { useSerialConnection } from "./hooks/useSerialConnection";
import {
  buildGoogleFormUrl,
  createTrialPlan,
  downloadCsv,
  formatFlowLabel,
  generateSeed,
  isRatingsComplete,
} from "./lib/experiment";
import type { ExperimentScreen, LikertKey, TrialRatings, TrialResult } from "./types";

const STEP_TITLES: Record<ExperimentScreen, string> = {
  welcome: "Welcome",
  connect: "Connect Device",
  participant: "Participant Setup",
  trial: "Trial",
  rating: "Rating",
  completion: "Completion",
};

function formatStatus(status: ReturnType<typeof useSerialConnection>["status"]): string {
  if (status === "connected") {
    return "Connected";
  }

  if (status === "connecting") {
    return "Connecting";
  }

  if (status === "unsupported") {
    return "Unsupported";
  }

  return "Disconnected";
}

function clampLikertValue(value: string): number | null {
  const parsed = Number(value);
  return LIKERT_VALUES.includes(parsed as (typeof LIKERT_VALUES)[number]) ? parsed : null;
}

function StudyCard({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-[32px] border border-white/70 bg-white/80 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.12)] backdrop-blur xl:p-10">
      <div className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">{eyebrow}</p>
        <h2 className="mt-3 text-3xl font-bold text-slate-950">{title}</h2>
        <p className="mt-3 max-w-xl text-sm leading-7 text-slate-500">{description}</p>
      </div>
      {children}
    </section>
  );
}

function StatusBadge({ tone, children }: { tone: "neutral" | "success" | "warning" | "danger"; children: ReactNode }) {
  const toneClasses = {
    neutral: "border-slate-200 bg-white text-slate-600",
    success: "border-emerald-200 bg-emerald-50 text-emerald-700",
    warning: "border-amber-200 bg-amber-50 text-amber-700",
    danger: "border-rose-200 bg-rose-50 text-rose-700",
  };

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold ${toneClasses[tone]}`}>
      {children}
    </span>
  );
}

function ProgressHeader({
  screen,
  participantId,
  completedTrials,
  currentTrialIndex,
  totalTrials,
  experimenterMode,
  sessionSeed,
  serialStatus,
  onToggleFullscreen,
  onReset,
  isFullscreen,
}: {
  screen: ExperimentScreen;
  participantId: string;
  completedTrials: number;
  currentTrialIndex: number;
  totalTrials: number;
  experimenterMode: boolean;
  sessionSeed: string;
  serialStatus: string;
  onToggleFullscreen: () => Promise<void>;
  onReset: () => void;
  isFullscreen: boolean;
}) {
  const trialNumber = Math.min(currentTrialIndex + 1, totalTrials);
  const progressValue =
    screen === "completion"
      ? 100
      : Math.min(100, Math.round((completedTrials / totalTrials) * 100));

  return (
    <header className="mb-6 rounded-[28px] border border-white/70 bg-white/75 px-6 py-5 shadow-[0_10px_40px_rgba(15,23,42,0.08)] backdrop-blur">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">{STEP_TITLES[screen]}</p>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <h1 className="text-xl font-bold text-slate-950">
              {screen === "completion" ? "Study complete" : `Trial ${trialNumber} of ${totalTrials}`}
            </h1>
            <StatusBadge tone={serialStatus === "Connected" ? "success" : serialStatus === "Unsupported" ? "danger" : "warning"}>
              {serialStatus}
            </StatusBadge>
          </div>
          <div className="mt-3 flex flex-wrap gap-3 text-sm text-slate-500">
            <span>Participant: {participantId || "Not set"}</span>
            <span>Completed: {completedTrials}/{totalTrials}</span>
            {experimenterMode ? <span>Experimenter mode on</span> : null}
            {sessionSeed ? <span>Seed: {sessionSeed}</span> : null}
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => void onToggleFullscreen()}
            className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
          >
            {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
            {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          </button>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-4 py-2 text-sm font-semibold text-rose-700 transition hover:bg-rose-100"
          >
            <RotateCcw className="h-4 w-4" />
            Reset Study
          </button>
        </div>
      </div>
      <div className="mt-5 h-2 rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-[linear-gradient(90deg,#0f766e_0%,#2563eb_60%,#7c3aed_100%)] transition-all duration-500"
          style={{ width: `${progressValue}%` }}
        />
      </div>
    </header>
  );
}

function LikertQuestionRow({
  index,
  prompt,
  value,
  isActive,
  onSelect,
  onFocus,
}: {
  index: number;
  prompt: string;
  value?: number;
  isActive: boolean;
  onSelect: (value: number) => void;
  onFocus: () => void;
}) {
  return (
    <div
      className={`rounded-[24px] border p-5 transition ${
        isActive ? "border-sky-200 bg-sky-50/60 shadow-sm" : "border-slate-200 bg-white"
      }`}
      onClick={onFocus}
    >
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Question {index + 1}</p>
          <p className="mt-2 text-sm font-medium leading-6 text-slate-800">{prompt}</p>
        </div>
        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-500">
          {value ? `Selected ${value}` : "1-7"}
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {LIKERT_VALUES.map((likertValue) => {
          const isSelected = value === likertValue;
          return (
            <button
              key={likertValue}
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onSelect(likertValue);
              }}
              className={`flex h-11 w-11 items-center justify-center rounded-full border text-sm font-semibold transition ${
                isSelected
                  ? "border-slate-900 bg-slate-900 text-white shadow-sm"
                  : "border-slate-200 bg-white text-slate-600 hover:border-slate-400 hover:text-slate-950"
              }`}
            >
              {likertValue}
            </button>
          );
        })}
      </div>
      <div className="mt-3 flex justify-between text-[11px] font-medium uppercase tracking-[0.2em] text-slate-400">
        <span>Low</span>
        <span>High</span>
      </div>
    </div>
  );
}

export function StudyApp() {
  const serial = useSerialConnection();
  const [screen, setScreen] = useState<ExperimentScreen>("welcome");
  const [participantIdInput, setParticipantIdInput] = useState("");
  const [participantId, setParticipantId] = useState("");
  const [seedInput, setSeedInput] = useState("");
  const [sessionSeed, setSessionSeed] = useState("");
  const [experimenterMode, setExperimenterMode] = useState(false);
  const [trialPlan, setTrialPlan] = useState(createTrialPlan(generateSeed()));
  const [currentTrialIndex, setCurrentTrialIndex] = useState(0);
  const [currentRatings, setCurrentRatings] = useState<TrialRatings>({});
  const [results, setResults] = useState<TrialResult[]>([]);
  const [trialPlayToken, setTrialPlayToken] = useState(0);
  const [ratingReplayToken, setRatingReplayToken] = useState(0);
  const [activeQuestionIndex, setActiveQuestionIndex] = useState(0);
  const [isTrialPlaying, setIsTrialPlaying] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(
    typeof document !== "undefined" ? Boolean(document.fullscreenElement) : false,
  );
  const [studyError, setStudyError] = useState<string | null>(null);
  const [sessionStartedAt, setSessionStartedAt] = useState<string | null>(null);
  const [sessionFinishedAt, setSessionFinishedAt] = useState<string | null>(null);

  const currentTrial = trialPlan[currentTrialIndex];
  const completedTrials = results.length;
  const googleFormsUrl = useMemo(
    () =>
      buildGoogleFormUrl(
        GOOGLE_FORM_CONFIG.googleFormUrl,
        GOOGLE_FORM_CONFIG.participantPrefillEntry,
        participantId,
      ),
    [participantId],
  );

  const serialStatusLabel = formatStatus(serial.status);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (screen !== "rating") {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      const activeElement = document.activeElement;
      const activeTag = activeElement instanceof HTMLElement ? activeElement.tagName : "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(activeTag)) {
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        setActiveQuestionIndex((index) => Math.min(index + 1, LIKERT_QUESTIONS.length - 1));
        return;
      }

      if (event.key === "ArrowUp") {
        event.preventDefault();
        setActiveQuestionIndex((index) => Math.max(index - 1, 0));
        return;
      }

      if (event.key === "Enter" && isRatingsComplete(currentRatings)) {
        event.preventDefault();
        void handleSubmitRatings();
        return;
      }

      const value = clampLikertValue(event.key);
      if (!value) {
        return;
      }

      event.preventDefault();
      const currentQuestionKey = LIKERT_QUESTIONS[activeQuestionIndex].key;
      updateRating(currentQuestionKey, value);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeQuestionIndex, currentRatings, screen]);

  useEffect(() => {
    if (serial.error) {
      setStudyError(serial.error);
    }
  }, [serial.error]);

  function getNextQuestionIndex(ratings: TrialRatings): number {
    const nextIndex = LIKERT_QUESTIONS.findIndex((question) => !ratings[question.key]);
    return nextIndex === -1 ? LIKERT_QUESTIONS.length - 1 : nextIndex;
  }

  function updateRating(questionKey: LikertKey, value: number) {
    setCurrentRatings((previous) => {
      const nextRatings = { ...previous, [questionKey]: value };
      setActiveQuestionIndex(getNextQuestionIndex(nextRatings));
      return nextRatings;
    });
  }

  async function handleToggleFullscreen() {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }

    await document.documentElement.requestFullscreen();
  }

  function resetStudy() {
    const shouldReset = window.confirm("Reset the study and clear the current participant session?");
    if (!shouldReset) {
      return;
    }

    startTransition(() => {
      setScreen("welcome");
      setParticipantIdInput("");
      setParticipantId("");
      setSeedInput("");
      setSessionSeed("");
      setTrialPlan(createTrialPlan(generateSeed()));
      setCurrentTrialIndex(0);
      setCurrentRatings({});
      setResults([]);
      setTrialPlayToken(0);
      setRatingReplayToken(0);
      setActiveQuestionIndex(0);
      setIsTrialPlaying(false);
      setStudyError(null);
      setSessionStartedAt(null);
      setSessionFinishedAt(null);
    });
  }

  async function handleConnectDevice() {
    const connected = await serial.connect();
    if (connected) {
      startTransition(() => setScreen("participant"));
    }
  }

  function handleBeginSetup() {
    setStudyError(null);
    startTransition(() => setScreen("connect"));
  }

  function handleCreateSession() {
    const normalizedParticipantId = participantIdInput.trim();

    if (!normalizedParticipantId) {
      setStudyError("Enter a participant ID before starting the study.");
      return;
    }

    const nextSeed = seedInput.trim() || generateSeed();
    const nextPlan = createTrialPlan(nextSeed);

    startTransition(() => {
      setParticipantId(normalizedParticipantId);
      setSessionSeed(nextSeed);
      setSeedInput(nextSeed);
      setTrialPlan(nextPlan);
      setCurrentTrialIndex(0);
      setResults([]);
      setCurrentRatings({});
      setActiveQuestionIndex(0);
      setTrialPlayToken(0);
      setRatingReplayToken(0);
      setSessionStartedAt(new Date().toISOString());
      setSessionFinishedAt(null);
      setStudyError(null);
      setScreen("trial");
    });
  }

  async function handleStartTrial() {
    if (!currentTrial) {
      return;
    }

    if (serial.status !== "connected") {
      setStudyError("Connect the ESP32 before starting a trial.");
      return;
    }

    setStudyError(null);
    setIsTrialPlaying(true);
    const didSend = await serial.sendStimulus(currentTrial.stimulusId);

    if (!didSend) {
      setIsTrialPlaying(false);
      return;
    }

    setTrialPlayToken((token) => token + 1);
  }

  function handleTrialPlaybackComplete() {
    startTransition(() => {
      setIsTrialPlaying(false);
      setCurrentRatings({});
      setActiveQuestionIndex(0);
      setRatingReplayToken(0);
      setScreen("rating");
    });
  }

  async function handleReplayStimulus() {
    if (!currentTrial) {
      return;
    }

    if (serial.status !== "connected") {
      setStudyError("Reconnect the ESP32 before replaying the haptic.");
      return;
    }

    setStudyError(null);
    const didSend = await serial.sendStimulus(currentTrial.stimulusId);
    if (didSend) {
      setRatingReplayToken((token) => token + 1);
    }
  }

  async function handleSubmitRatings() {
    if (!currentTrial) {
      return;
    }

    if (!isRatingsComplete(currentRatings)) {
      setStudyError("Complete all five rating questions before continuing.");
      return;
    }

    const ratingSnapshot = currentRatings as Record<LikertKey, number>;
    const nextResult: TrialResult = {
      ...currentTrial,
      participantId,
      ...ratingSnapshot,
      timestamp: new Date().toISOString(),
    };

    const nextResults = [...results, nextResult];
    const hasMoreTrials = currentTrialIndex < trialPlan.length - 1;

    startTransition(() => {
      setResults(nextResults);
      setCurrentRatings({});
      setActiveQuestionIndex(0);
      setStudyError(null);
      setRatingReplayToken(0);

      if (hasMoreTrials) {
        setCurrentTrialIndex((index) => index + 1);
        setScreen("trial");
      } else {
        setSessionFinishedAt(nextResult.timestamp);
        setScreen("completion");
      }
    });
  }

  function renderWelcomeScreen() {
    return (
      <StudyCard
        eyebrow="Panel of Stimuli"
        title="Local haptic thesis prototype"
        description="Run the participant study locally, pair the browser with your ESP32 over Web Serial, present all 60 trials, and export clean CSV data with no backend."
      >
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-4 text-sm leading-7 text-slate-600">
            <div className="rounded-[24px] border border-slate-200 bg-slate-50/80 p-5">
              <p className="font-semibold text-slate-900">Study flow</p>
              <p className="mt-2">
                Connect device, enter participant ID, run 60 randomized trials, capture five Likert ratings,
                export CSV, then open Google Forms for any end-of-study follow-up.
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-[22px] border border-emerald-100 bg-emerald-50/75 p-4">
                <p className="font-semibold text-emerald-900">4 UI flows</p>
                <p className="mt-2 text-sm text-emerald-800">Success, error, notification, and loading.</p>
              </div>
              <div className="rounded-[22px] border border-sky-100 bg-sky-50/75 p-4">
                <p className="font-semibold text-sky-900">15 stimuli</p>
                <p className="mt-2 text-sm text-sky-800">Triggered as simple serial integers `1-15`.</p>
              </div>
            </div>
            <div className="rounded-[24px] border border-amber-100 bg-amber-50/80 p-5">
              <p className="flex items-center gap-2 font-semibold text-amber-900">
                <ShieldEllipsis className="h-4 w-4" />
                Researcher note
              </p>
              <p className="mt-2">
                Use the latest Chrome or Edge and launch the app from `http://localhost` so Web Serial is available.
              </p>
            </div>
          </div>

          <div className="rounded-[28px] border border-white/70 bg-[linear-gradient(135deg,rgba(255,255,255,0.96),rgba(241,245,249,0.92))] p-5 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Visual Reference</p>
                <h3 className="mt-2 text-lg font-bold text-slate-900">Success flow preview</h3>
              </div>
              <Sparkles className="h-5 w-5 text-emerald-500" />
            </div>
            <FlowRenderer uiFlow="success" />
          </div>
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleBeginSetup}
            className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
          >
            Begin Setup
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </StudyCard>
    );
  }

  function renderConnectScreen() {
    const tone =
      serial.status === "connected" ? "success" : serial.status === "unsupported" ? "danger" : "warning";

    return (
      <StudyCard
        eyebrow="Step 1"
        title="Connect the ESP32"
        description="The browser will send a newline-terminated stimulus ID such as `7` to the ESP32. Once connected, the device stays available for the rest of the session unless you disconnect it."
      >
        <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-4">
            <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-6">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white shadow-sm">
                  <Usb className="h-5 w-5 text-slate-700" />
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Device status</p>
                  <div className="mt-2 flex items-center gap-3">
                    <StatusBadge tone={tone}>{serialStatusLabel}</StatusBadge>
                    {serial.portLabel ? <span className="text-sm text-slate-500">{serial.portLabel}</span> : null}
                  </div>
                </div>
              </div>

              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="rounded-[20px] border border-white bg-white p-4 shadow-sm">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Protocol</p>
                  <p className="mt-2 text-sm font-medium text-slate-900">`stimulus_id\n`</p>
                  <p className="mt-1 text-xs text-slate-500">Example: `7\n`</p>
                </div>
                <div className="rounded-[20px] border border-white bg-white p-4 shadow-sm">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Baud rate</p>
                  <p className="mt-2 text-sm font-medium text-slate-900">115200</p>
                  <p className="mt-1 text-xs text-slate-500">ESP32 + DRV2605L playback</p>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void handleConnectDevice()}
                disabled={serial.status === "connecting" || !serial.isSupported}
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                <Cable className="h-4 w-4" />
                {serial.status === "connected" ? "Reconnect Device" : "Connect Device"}
              </button>

              {serial.status === "connected" ? (
                <>
                  <button
                    type="button"
                    onClick={() => startTransition(() => setScreen("participant"))}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
                  >
                    Continue
                    <ChevronRight className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => void serial.disconnect()}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
                  >
                    Disconnect
                  </button>
                </>
              ) : null}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.95),rgba(239,246,255,0.8))] p-5 shadow-sm">
            <FlowRenderer uiFlow="notification" />
          </div>
        </div>
      </StudyCard>
    );
  }

  function renderParticipantScreen() {
    return (
      <StudyCard
        eyebrow="Step 2"
        title="Participant setup"
        description="Assign the participant ID, optionally pin a reproducible seed, and choose whether to reveal experimenter metadata such as the hidden stimulus ID."
      >
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-5">
            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-slate-700">Participant ID</span>
              <input
                value={participantIdInput}
                onChange={(event) => setParticipantIdInput(event.target.value)}
                className="w-full rounded-[20px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-slate-400 focus:ring-4 focus:ring-slate-100"
                placeholder="e.g. P-014"
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-semibold text-slate-700">Random seed</span>
              <input
                value={seedInput}
                onChange={(event) => setSeedInput(event.target.value)}
                className="w-full rounded-[20px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-slate-400 focus:ring-4 focus:ring-slate-100"
                placeholder="Leave blank to auto-generate"
              />
              <p className="mt-2 text-xs text-slate-500">Use the same seed to reproduce the same 60-trial order.</p>
            </label>

            <label className="flex items-start gap-4 rounded-[24px] border border-slate-200 bg-slate-50/80 p-5">
              <input
                type="checkbox"
                checked={experimenterMode}
                onChange={(event) => setExperimenterMode(event.target.checked)}
                className="mt-1 h-5 w-5 rounded border-slate-300 text-slate-900 focus:ring-slate-300"
              />
              <div>
                <p className="text-sm font-semibold text-slate-900">Experimenter mode</p>
                <p className="mt-2 text-sm leading-6 text-slate-500">
                  Reveal the randomized seed, UI flow labels, and current stimulus IDs during the session.
                </p>
              </div>
            </label>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={handleCreateSession}
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
              >
                Start Study
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="space-y-4 rounded-[28px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.96),rgba(239,246,255,0.82))] p-6 shadow-sm">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Study design</p>
              <h3 className="mt-2 text-lg font-bold text-slate-900">60 total trials</h3>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-[20px] border border-slate-100 bg-white p-4">
                <p className="text-sm font-semibold text-slate-900">4 UI flows</p>
                <p className="mt-2 text-sm text-slate-500">Each flow is paired with all 15 haptic stimuli.</p>
              </div>
              <div className="rounded-[20px] border border-slate-100 bg-white p-4">
                <p className="text-sm font-semibold text-slate-900">5 rating questions</p>
                <p className="mt-2 text-sm text-slate-500">Likert scale from 1 to 7 for every trial.</p>
              </div>
            </div>
            <div className="rounded-[24px] border border-sky-100 bg-sky-50/80 p-5">
              <p className="text-sm font-semibold text-sky-900">Current serial status</p>
              <p className="mt-2 text-sm text-sky-800">{serialStatusLabel}</p>
            </div>
            <FlowRenderer uiFlow="error" />
          </div>
        </div>
      </StudyCard>
    );
  }

  function renderTrialScreen() {
    if (!currentTrial) {
      return null;
    }

    return (
      <StudyCard
        eyebrow={`Trial ${currentTrial.trialIndex}`}
        title={formatFlowLabel(currentTrial.uiFlow)}
        description="Ask the participant to watch the UI event and feel the paired haptic. The animation and ESP32 playback will start together."
      >
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-4">
            <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-6">
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Instructions</p>
              <ol className="mt-4 space-y-3 text-sm leading-6 text-slate-600">
                <li>1. Confirm the participant is ready.</li>
                <li>2. Press start to trigger the UI flow and the haptic stimulus.</li>
                <li>3. Let the animation finish before moving to the rating screen.</li>
              </ol>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void handleStartTrial()}
                disabled={isTrialPlaying}
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                <Play className="h-4 w-4" />
                {isTrialPlaying ? "Playing..." : "Start Trial"}
              </button>
              <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-4 py-3 text-sm text-slate-500">
                Playback window: {Math.round(FLOW_PLAYBACK_MS / 100) / 10}s
              </span>
            </div>

            {experimenterMode ? (
              <div className="rounded-[24px] border border-amber-200 bg-amber-50 p-5">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-700">Experimenter mode</p>
                <div className="mt-3 grid gap-3 text-sm text-amber-900 sm:grid-cols-2">
                  <p>Stimulus ID: {currentTrial.stimulusId}</p>
                  <p>UI flow: {currentTrial.uiFlow}</p>
                  <p>Seed: {sessionSeed}</p>
                  <p>Trial index: {currentTrial.trialIndex}</p>
                </div>
              </div>
            ) : null}
          </div>

          <div className="rounded-[28px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.96),rgba(248,250,252,0.92))] p-5 shadow-sm">
            <FlowRenderer
              uiFlow={currentTrial.uiFlow}
              playToken={trialPlayToken}
              onPlaybackComplete={handleTrialPlaybackComplete}
            />
          </div>
        </div>
      </StudyCard>
    );
  }

  function renderRatingScreen() {
    if (!currentTrial) {
      return null;
    }

    return (
      <StudyCard
        eyebrow={`Rate Trial ${currentTrial.trialIndex}`}
        title="Participant rating"
        description="Capture five Likert ratings from 1 to 7. Keyboard shortcuts are enabled: number keys score the active question, arrow keys move between questions, and Enter submits when complete."
      >
        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <div className="space-y-4">
            <div className="rounded-[28px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.96),rgba(239,246,255,0.82))] p-5 shadow-sm">
              <FlowRenderer uiFlow={currentTrial.uiFlow} playToken={ratingReplayToken} />
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void handleReplayStimulus()}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950"
              >
                <RefreshCcw className="h-4 w-4" />
                Replay Stimulus
              </button>
              {experimenterMode ? (
                <span className="inline-flex items-center rounded-full border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800">
                  Stimulus ID {currentTrial.stimulusId}
                </span>
              ) : null}
            </div>
          </div>

          <div className="space-y-4">
            {LIKERT_QUESTIONS.map((question, index) => (
              <LikertQuestionRow
                key={question.key}
                index={index}
                prompt={question.prompt}
                value={currentRatings[question.key]}
                isActive={activeQuestionIndex === index}
                onSelect={(value) => updateRating(question.key, value)}
                onFocus={() => setActiveQuestionIndex(index)}
              />
            ))}

            <div className="flex flex-wrap gap-3 pt-2">
              <button
                type="button"
                onClick={() => void handleSubmitRatings()}
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
              >
                Continue to Next Trial
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </StudyCard>
    );
  }

  function renderCompletionScreen() {
    return (
      <StudyCard
        eyebrow="Session complete"
        title="Export and wrap up"
        description="The participant session is finished. Export the CSV immediately, then open the follow-up Google Form if you want to collect any end-of-study responses."
      >
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-4">
            <div className="rounded-[28px] border border-emerald-100 bg-emerald-50/80 p-6">
              <p className="text-sm font-semibold text-emerald-900">Session summary</p>
              <div className="mt-4 grid gap-3 text-sm text-emerald-950 sm:grid-cols-2">
                <p>Participant: {participantId}</p>
                <p>Trials completed: {results.length}/{TOTAL_TRIALS}</p>
                <p>Seed: {sessionSeed}</p>
                <p>Finished: {sessionFinishedAt ?? "N/A"}</p>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => downloadCsv(results, participantId, sessionSeed)}
                className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
              >
                <Download className="h-4 w-4" />
                Export CSV
              </button>
              <button
                type="button"
                onClick={() => googleFormsUrl && window.open(googleFormsUrl, "_blank", "noopener,noreferrer")}
                disabled={!googleFormsUrl}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:text-slate-950 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300"
              >
                <ExternalLink className="h-4 w-4" />
                Open Google Form
              </button>
            </div>

            {!googleFormsUrl ? (
              <div className="rounded-[24px] border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-900">
                Configure `googleFormUrl` and `participantPrefillEntry` in `src/app/config.ts` to enable the Google Forms hand-off.
              </div>
            ) : null}
          </div>

          <div className="rounded-[28px] border border-white/80 bg-[linear-gradient(145deg,rgba(255,255,255,0.96),rgba(240,249,255,0.86))] p-5 shadow-sm">
            <FlowRenderer uiFlow="loading" playToken={1} />
          </div>
        </div>
      </StudyCard>
    );
  }

  function renderCurrentScreen() {
    if (screen === "welcome") {
      return renderWelcomeScreen();
    }

    if (screen === "connect") {
      return renderConnectScreen();
    }

    if (screen === "participant") {
      return renderParticipantScreen();
    }

    if (screen === "trial") {
      return renderTrialScreen();
    }

    if (screen === "rating") {
      return renderRatingScreen();
    }

    return renderCompletionScreen();
  }

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(125,211,252,0.24),_transparent_32%),linear-gradient(180deg,#f8fafc_0%,#eef2ff_32%,#f8fafc_100%)] px-4 py-6 text-slate-950 sm:px-6 lg:px-8 lg:py-8">
      <div className="mx-auto max-w-7xl">
        {screen !== "welcome" ? (
          <ProgressHeader
            screen={screen}
            participantId={participantId}
            completedTrials={completedTrials}
            currentTrialIndex={currentTrialIndex}
            totalTrials={TOTAL_TRIALS}
            experimenterMode={experimenterMode}
            sessionSeed={sessionSeed}
            serialStatus={serialStatusLabel}
            onToggleFullscreen={handleToggleFullscreen}
            onReset={resetStudy}
            isFullscreen={isFullscreen}
          />
        ) : null}

        {studyError ? (
          <div className="mb-6 flex items-start gap-3 rounded-[24px] border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-900 shadow-sm">
            <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0" />
            <div>
              <p className="font-semibold">Study warning</p>
              <p className="mt-1 leading-6">{studyError}</p>
            </div>
          </div>
        ) : null}

        {renderCurrentScreen()}

        <footer className="mt-6 px-1 text-xs leading-6 text-slate-400">
          {sessionStartedAt ? `Session started: ${sessionStartedAt}. ` : ""}
          Browser must run locally over HTTP(S) for Web Serial access.
        </footer>
      </div>
    </div>
  );
}
