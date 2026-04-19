import { useEffect, useState } from "react";
import {
  Check,
  ChevronRight,
  Download,
  ExternalLink,
  Maximize2,
  Minimize2,
  RefreshCcw,
  RotateCcw,
  Usb,
} from "lucide-react";

import {
  FLOW_ORDER,
  GOOGLE_FORM_CONFIG,
  LIKERT_QUESTIONS,
  LIKERT_VALUES,
  STIMULI_PER_FLOW,
  TOTAL_ANON_STIMULI,
} from "./config";
import { FlowRenderer } from "./components/FlowRenderer";
import { useSerialConnection } from "./hooks/useSerialConnection";
import {
  buildAnonymousIdToWaveformSlot,
  buildBlockResultsCsv,
  buildGoogleFormUrl,
  buildOverviewCsv,
  createFlowBlocks,
  downloadTextFile,
  formatFlowLabel,
  formatFlowShortLabel,
  generateSeed,
  isRatingsComplete,
} from "./lib/experiment";
import type {
  ExperimentScreen,
  FlowBlockResult,
  FlowOverviewResult,
  FlowRatings,
  FlowStimulusDefinition,
  LikertKey,
  UIFlow,
} from "./types";

interface OverviewDraft {
  top3Ids: number[];
  bottomIds: number[];
  topReason: string;
  bottomReason: string;
}

const EMPTY_OVERVIEW_DRAFT: OverviewDraft = {
  top3Ids: [],
  bottomIds: [],
  topReason: "",
  bottomReason: "",
};

const ANONYMOUS_ID_TO_WAVEFORM_SLOT = buildAnonymousIdToWaveformSlot();

function getStatusLabel(status: ReturnType<typeof useSerialConnection>["status"]): string {
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

function getStatusTone(status: ReturnType<typeof useSerialConnection>["status"]): string {
  if (status === "connected") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700";
  }

  if (status === "connecting") {
    return "border-amber-200 bg-amber-50 text-amber-700";
  }

  if (status === "unsupported") {
    return "border-rose-200 bg-rose-50 text-rose-700";
  }

  return "border-slate-200 bg-white text-slate-600";
}

function createDefaultOverviewDraft(): OverviewDraft {
  return {
    top3Ids: [],
    bottomIds: [],
    topReason: "",
    bottomReason: "",
  };
}

function sanitizeParticipantId(value: string): string {
  return value.trim().replace(/\s+/g, "-");
}

function HeaderBar({
  currentFlow,
  deviceLabel,
  experimenterMode,
  isFullscreen,
  onReset,
  onToggleFullscreen,
  overallCompletedCount,
  overviewCompletedCount,
  participantId,
  screen,
  seed,
  serialStatus,
}: {
  currentFlow: UIFlow | null;
  deviceLabel: string | null;
  experimenterMode: boolean;
  isFullscreen: boolean;
  onReset: () => void;
  onToggleFullscreen: () => void;
  overallCompletedCount: number;
  overviewCompletedCount: number;
  participantId: string;
  screen: ExperimentScreen;
  seed: string;
  serialStatus: ReturnType<typeof useSerialConnection>["status"];
}) {
  const totalProgressSteps = TOTAL_ANON_STIMULI + FLOW_ORDER.length;
  const progressValue = ((overallCompletedCount + overviewCompletedCount) / totalProgressSteps) * 100;

  return (
    <div className="sticky top-0 z-30 border-b border-white/70 bg-white/85 backdrop-blur">
      <div className="mx-auto max-w-[1500px] px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                Panel of Stimuli
              </p>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-slate-500">
                <span>
                  Participant:{" "}
                  <span className="font-semibold text-slate-900">{participantId || "Not set"}</span>
                </span>
                <span>
                  Screen: <span className="font-semibold text-slate-900">{screen}</span>
                </span>
                {currentFlow ? (
                  <span>
                    Flow: <span className="font-semibold text-slate-900">{formatFlowShortLabel(currentFlow)}</span>
                  </span>
                ) : null}
                {experimenterMode ? (
                  <span>
                    Seed: <span className="font-semibold text-slate-900">{seed || "Auto"}</span>
                  </span>
                ) : null}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <div className={`rounded-full border px-3 py-1.5 text-sm font-medium ${getStatusTone(serialStatus)}`}>
                <span className="inline-flex items-center gap-2">
                  <Usb className="h-4 w-4" />
                  {getStatusLabel(serialStatus)}
                  {deviceLabel ? ` · ${deviceLabel}` : ""}
                </span>
              </div>
              <button
                type="button"
                onClick={onToggleFullscreen}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50"
              >
                {isFullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                {isFullscreen ? "Exit fullscreen" : "Fullscreen"}
              </button>
              <button
                type="button"
                onClick={onReset}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm text-slate-500">
              <span>
                Rated vibrations {overallCompletedCount}/{TOTAL_ANON_STIMULI}
              </span>
              <span>
                Flow overviews {overviewCompletedCount}/{FLOW_ORDER.length}
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-gradient-to-r from-slate-900 via-sky-700 to-cyan-600 transition-[width] duration-500"
                style={{ width: `${progressValue}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LikertQuestionRow({
  disabled,
  prompt,
  selectedValue,
  onSelect,
}: {
  disabled: boolean;
  prompt: string;
  selectedValue?: number;
  onSelect: (value: number) => void;
}) {
  return (
    <div className="rounded-[28px] border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-4">
        <div className="space-y-1">
          <p className="text-base font-semibold text-slate-950">{prompt}</p>
          <div className="flex items-center justify-between text-xs uppercase tracking-[0.18em] text-slate-400">
            <span>Low</span>
            <span>7-point Likert</span>
            <span>High</span>
          </div>
        </div>
        <div className="grid grid-cols-7 gap-2">
          {LIKERT_VALUES.map((value) => {
            const isSelected = value === selectedValue;
            return (
              <button
                key={value}
                type="button"
                disabled={disabled}
                onClick={() => onSelect(value)}
                className={`rounded-2xl border px-0 py-3 text-sm font-semibold transition ${
                  isSelected
                    ? "border-slate-900 bg-slate-900 text-white shadow-sm"
                    : "border-slate-200 bg-slate-50 text-slate-600 hover:border-slate-300 hover:bg-slate-100"
                } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
              >
                {value}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export function StudyApp() {
  const serial = useSerialConnection();
  const [screen, setScreen] = useState<ExperimentScreen>("welcome");
  const [participantIdInput, setParticipantIdInput] = useState("");
  const [participantId, setParticipantId] = useState("");
  const [seedInput, setSeedInput] = useState(generateSeed());
  const [seed, setSeed] = useState("");
  const [experimenterMode, setExperimenterMode] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(Boolean(document.fullscreenElement));
  const [flowBlocks, setFlowBlocks] = useState<Record<UIFlow, FlowStimulusDefinition[]> | null>(null);
  const [currentFlowIndex, setCurrentFlowIndex] = useState(0);
  const [selectedStimulusId, setSelectedStimulusId] = useState<number | null>(null);
  const [playToken, setPlayToken] = useState(0);
  const [isPlaybackRunning, setIsPlaybackRunning] = useState(false);
  const [studyMessage, setStudyMessage] = useState<string | null>(null);
  const [ratingsByStimulusId, setRatingsByStimulusId] = useState<Record<number, FlowRatings>>({});
  const [blockResultsByStimulusId, setBlockResultsByStimulusId] = useState<Record<number, FlowBlockResult>>({});
  const [overviewDrafts, setOverviewDrafts] = useState<Partial<Record<UIFlow, OverviewDraft>>>({});
  const [overviewResults, setOverviewResults] = useState<Partial<Record<UIFlow, FlowOverviewResult>>>({});

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const currentFlow = screen === "flow-block" || screen === "flow-overview" ? FLOW_ORDER[currentFlowIndex] : null;
  const currentStimuli = currentFlow && flowBlocks ? flowBlocks[currentFlow] : [];
  const currentStimulus =
    selectedStimulusId && currentStimuli ? currentStimuli.find((item) => item.anonymousId === selectedStimulusId) ?? null : null;
  const currentRatings = currentStimulus ? ratingsByStimulusId[currentStimulus.anonymousId] ?? {} : {};
  const currentFlowCompletedCount = currentStimuli.filter((item) => Boolean(blockResultsByStimulusId[item.anonymousId])).length;
  const overallCompletedCount = Object.values(blockResultsByStimulusId).length;
  const overviewCompletedCount = Object.values(overviewResults).filter(Boolean).length;
  const currentOverviewDraft = currentFlow ? overviewDrafts[currentFlow] ?? EMPTY_OVERVIEW_DRAFT : EMPTY_OVERVIEW_DRAFT;

  function handleReset() {
    setScreen("welcome");
    setParticipantIdInput("");
    setParticipantId("");
    setSeedInput(generateSeed());
    setSeed("");
    setExperimenterMode(false);
    setFlowBlocks(null);
    setCurrentFlowIndex(0);
    setSelectedStimulusId(null);
    setPlayToken(0);
    setIsPlaybackRunning(false);
    setStudyMessage(null);
    setRatingsByStimulusId({});
    setBlockResultsByStimulusId({});
    setOverviewDrafts({});
    setOverviewResults({});
    void serial.disconnect();
  }

  async function handleToggleFullscreen() {
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }

    await document.documentElement.requestFullscreen();
  }

  function handleStartSession() {
    const nextParticipantId = sanitizeParticipantId(participantIdInput);
    const nextSeed = seedInput.trim() || generateSeed();

    if (!nextParticipantId) {
      setStudyMessage("Enter a participant ID before starting the study.");
      return;
    }

    const nextBlocks = createFlowBlocks(nextSeed);

    setParticipantId(nextParticipantId);
    setSeed(nextSeed);
    setSeedInput(nextSeed);
    setFlowBlocks(nextBlocks);
    setCurrentFlowIndex(0);
    setSelectedStimulusId(null);
    setPlayToken(0);
    setRatingsByStimulusId({});
    setBlockResultsByStimulusId({});
    setOverviewDrafts({});
    setOverviewResults({});
    setStudyMessage(null);
    setScreen("flow-block");
  }

  async function triggerStimulusPlayback(stimulus: FlowStimulusDefinition) {
    setSelectedStimulusId(stimulus.anonymousId);
    setPlayToken((value) => value + 1);
    setIsPlaybackRunning(true);

    if (serial.status !== "connected") {
      setStudyMessage("Device not connected. The UI demo is still shown, but no haptic is sent.");
      return;
    }

    const sent = await serial.sendStimulus(stimulus.anonymousId);

    if (!sent) {
      setStudyMessage("The UI demo is still shown, but the anonymous vibration ID did not reach the ESP32.");
      return;
    }

    setStudyMessage(`Triggered ${stimulus.displayLabel} for ${formatFlowShortLabel(stimulus.flow)}.`);
  }

  function handlePlaybackComplete() {
    setIsPlaybackRunning(false);
  }

  function handleRatingSelect(key: LikertKey, value: number) {
    if (!currentStimulus) {
      return;
    }

    const nextRatings = {
      ...(ratingsByStimulusId[currentStimulus.anonymousId] ?? {}),
      [key]: value,
    };

    setRatingsByStimulusId((previous) => ({
      ...previous,
      [currentStimulus.anonymousId]: nextRatings,
    }));

    if (!isRatingsComplete(nextRatings)) {
      return;
    }

    setBlockResultsByStimulusId((previous) => ({
      ...previous,
      [currentStimulus.anonymousId]: {
        participantId,
        flow: currentStimulus.flow,
        anonymousId: currentStimulus.anonymousId,
        waveformSlot: currentStimulus.waveformSlot,
        gridPosition: currentStimulus.gridPosition,
        rating_match: nextRatings.rating_match!,
        rating_appropriate: nextRatings.rating_appropriate!,
        rating_meaningful: nextRatings.rating_meaningful!,
        timestamp: new Date().toISOString(),
      },
    }));
  }

  function handleContinueToOverview() {
    if (currentFlowCompletedCount < STIMULI_PER_FLOW) {
      setStudyMessage("All 15 anonymous vibrations in this block must be completed before the overview step.");
      return;
    }

    if (currentFlow) {
      setOverviewDrafts((previous) => ({
        ...previous,
        [currentFlow]: previous[currentFlow] ?? createDefaultOverviewDraft(),
      }));
    }

    setStudyMessage(null);
    setScreen("flow-overview");
  }

  function handleToggleOverviewSelection(bucket: "top3Ids" | "bottomIds", anonymousId: number) {
    if (!currentFlow) {
      return;
    }

    setOverviewDrafts((previous) => {
      const currentDraft = previous[currentFlow] ?? createDefaultOverviewDraft();
      const targetList = currentDraft[bucket];
      const otherBucket = bucket === "top3Ids" ? "bottomIds" : "top3Ids";
      const otherList = currentDraft[otherBucket];

      const isSelected = targetList.includes(anonymousId);
      const maxCount = bucket === "top3Ids" ? 3 : 2;
      let nextTarget = targetList;

      if (isSelected) {
        nextTarget = targetList.filter((value) => value !== anonymousId);
      } else {
        if (targetList.length >= maxCount) {
          return previous;
        }

        nextTarget = [...targetList, anonymousId];
      }

      const nextOther = otherList.filter((value) => value !== anonymousId);

      return {
        ...previous,
        [currentFlow]: {
          ...currentDraft,
          [bucket]: nextTarget,
          [otherBucket]: nextOther,
        },
      };
    });
  }

  function handleOverviewTextChange(field: "topReason" | "bottomReason", value: string) {
    if (!currentFlow) {
      return;
    }

    setOverviewDrafts((previous) => ({
      ...previous,
      [currentFlow]: {
        ...(previous[currentFlow] ?? createDefaultOverviewDraft()),
        [field]: value,
      },
    }));
  }

  function handleSaveOverview() {
    if (!currentFlow) {
      return;
    }

    const draft = overviewDrafts[currentFlow] ?? createDefaultOverviewDraft();

    if (draft.top3Ids.length !== 3) {
      setStudyMessage("Select exactly three best-match anonymous IDs before continuing.");
      return;
    }

    if (draft.bottomIds.length < 1 || draft.bottomIds.length > 2) {
      setStudyMessage("Select one or two worst-match anonymous IDs before continuing.");
      return;
    }

    setOverviewResults((previous) => ({
      ...previous,
      [currentFlow]: {
        participantId,
        flow: currentFlow,
        top3Ids: draft.top3Ids,
        bottomIds: draft.bottomIds,
        topReason: draft.topReason.trim(),
        bottomReason: draft.bottomReason.trim(),
        timestamp: new Date().toISOString(),
      },
    }));

    setStudyMessage(null);
    setSelectedStimulusId(null);

    if (currentFlowIndex === FLOW_ORDER.length - 1) {
      setScreen("completion");
      return;
    }

    const nextFlowIndex = currentFlowIndex + 1;
    setCurrentFlowIndex(nextFlowIndex);
    setSelectedStimulusId(null);
    setScreen("flow-block");
  }

  function handleDownloadBlockResults() {
    const filename = `${participantId || "participant"}_panel_block_ratings.csv`;
    downloadTextFile(buildBlockResultsCsv(Object.values(blockResultsByStimulusId)), filename);
  }

  function handleDownloadOverviewResults() {
    const filename = `${participantId || "participant"}_panel_overview.csv`;
    const orderedResults = FLOW_ORDER.flatMap((flow) => (overviewResults[flow] ? [overviewResults[flow]!] : []));
    downloadTextFile(buildOverviewCsv(orderedResults), filename);
  }

  const googleFormsUrl = buildGoogleFormUrl(
    GOOGLE_FORM_CONFIG.googleFormUrl,
    GOOGLE_FORM_CONFIG.participantPrefillEntry,
    participantId,
  );

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,rgba(226,247,255,0.9),transparent_35%),linear-gradient(180deg,#f8fbff_0%,#edf4fb_45%,#f8fafc_100%)] text-slate-950">
      {screen !== "welcome" ? (
        <HeaderBar
          currentFlow={currentFlow}
          deviceLabel={serial.portLabel}
          experimenterMode={experimenterMode}
          isFullscreen={isFullscreen}
          onReset={handleReset}
          onToggleFullscreen={handleToggleFullscreen}
          overallCompletedCount={overallCompletedCount}
          overviewCompletedCount={overviewCompletedCount}
          participantId={participantId}
          screen={screen}
          seed={seed}
          serialStatus={serial.status}
        />
      ) : null}

      <main className="mx-auto max-w-[1500px] px-4 py-8 sm:px-6 lg:px-8">
        {studyMessage ? (
          <div className="mb-6 rounded-[24px] border border-sky-200 bg-white/90 px-5 py-4 text-sm text-slate-700 shadow-sm">
            {studyMessage}
          </div>
        ) : null}
        {serial.error ? (
          <div className="mb-6 rounded-[24px] border border-rose-200 bg-rose-50/90 px-5 py-4 text-sm text-rose-700 shadow-sm">
            {serial.error}
          </div>
        ) : null}

        {screen === "welcome" ? (
          <section className="mx-auto max-w-5xl rounded-[40px] border border-white/70 bg-white/85 p-8 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur sm:p-10 lg:p-14">
            <div className="grid gap-10 lg:grid-cols-[1.2fr_0.8fr] lg:items-center">
              <div className="space-y-6">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.26em] text-sky-600">Local Research Prototype</p>
                  <h1 className="mt-4 max-w-3xl text-4xl font-bold leading-tight text-slate-950 sm:text-5xl">
                    Panel of Stimuli
                  </h1>
                  <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
                    Compare 15 anonymous haptic stimuli inside each UI event block, keep the UI flow fixed in place,
                    and export structured ratings plus per-flow Top/Bottom choices.
                  </p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  {FLOW_ORDER.map((flow) => (
                    <div key={flow} className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-5">
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                        {formatFlowShortLabel(flow)}
                      </p>
                      <p className="mt-3 text-sm leading-6 text-slate-600">{formatFlowLabel(flow)}</p>
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={() => {
                    setStudyMessage(null);
                    setScreen("connect");
                  }}
                  className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800"
                >
                  Start setup
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>

              <div className="rounded-[36px] border border-slate-200 bg-slate-50/80 p-6">
                <div className="space-y-4">
                  <div className="rounded-[28px] border border-white bg-white p-5 shadow-sm">
                    <p className="text-sm font-semibold text-slate-900">Study structure</p>
                    <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-600">
                      <li>4 UI flow blocks in a fixed order</li>
                      <li>15 anonymous vibration IDs per block</li>
                      <li>3 ratings per selected stimulus</li>
                      <li>Per-flow Top 3 and Bottom 1-2 overview</li>
                    </ul>
                  </div>
                  <div className="rounded-[28px] border border-white bg-white p-5 shadow-sm">
                    <p className="text-sm font-semibold text-slate-900">Serial contract</p>
                    <p className="mt-3 text-sm leading-6 text-slate-600">
                      The browser sends an anonymous ID like <code>27\n</code> to the ESP32, which then maps it to the
                      underlying waveform slot.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {screen === "connect" ? (
          <section className="mx-auto max-w-4xl rounded-[40px] border border-white/70 bg-white/85 p-8 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur sm:p-10">
            <div className="grid gap-8 lg:grid-cols-[1fr_0.8fr]">
              <div className="space-y-5">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Step 1</p>
                  <h2 className="mt-3 text-3xl font-bold text-slate-950">Connect the ESP32</h2>
                  <p className="mt-3 max-w-xl text-base leading-7 text-slate-600">
                    Use Chrome or Edge, then pair the Web Serial device. The app sends anonymous IDs 1-60 and leaves the
                    waveform mapping to the ESP32 sketch.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => void serial.connect()}
                    disabled={serial.status === "connecting"}
                    className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    <Usb className="h-4 w-4" />
                    {serial.status === "connected" ? "Reconnect device" : "Connect device"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void serial.disconnect()}
                    disabled={serial.status !== "connected"}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    Disconnect
                  </button>
                </div>
              </div>

              <div className="space-y-4 rounded-[32px] border border-slate-200 bg-slate-50/90 p-6">
                <div className={`rounded-[22px] border px-4 py-4 text-sm font-medium ${getStatusTone(serial.status)}`}>
                  Device status: {getStatusLabel(serial.status)}
                </div>
                <div className="rounded-[22px] border border-white bg-white px-4 py-4 text-sm leading-6 text-slate-600">
                  <p>Baud rate: 115200</p>
                  <p>Payload: anonymous ID plus newline</p>
                  <p>Example: <code>27\n</code></p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setStudyMessage(serial.status === "connected" ? null : "Continuing with setup. Playback will only send haptics after the device is connected.");
                    setScreen("participant");
                  }}
                  className="inline-flex items-center gap-2 rounded-full bg-sky-600 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-sky-500"
                >
                  Continue to participant setup
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </section>
        ) : null}

        {screen === "participant" ? (
          <section className="mx-auto max-w-4xl rounded-[40px] border border-white/70 bg-white/85 p-8 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur sm:p-10">
            <div className="grid gap-8 lg:grid-cols-[1fr_0.8fr]">
              <div className="space-y-6">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Step 2</p>
                  <h2 className="mt-3 text-3xl font-bold text-slate-950">Participant setup</h2>
                  <p className="mt-3 max-w-xl text-base leading-7 text-slate-600">
                    Each participant gets the same four flow blocks, with a participant-specific random layout for the 15
                    anonymous vibration cards inside each block.
                  </p>
                </div>

                <div className="space-y-4">
                  <label className="block">
                    <span className="mb-2 block text-sm font-semibold text-slate-700">Participant ID</span>
                    <input
                      value={participantIdInput}
                      onChange={(event) => setParticipantIdInput(event.target.value)}
                      placeholder="e.g. P01"
                      className="w-full rounded-[20px] border border-slate-200 bg-white px-4 py-3 text-slate-900 shadow-sm outline-none transition focus:border-slate-400"
                    />
                  </label>

                  <label className="block">
                    <span className="mb-2 block text-sm font-semibold text-slate-700">Random seed</span>
                    <input
                      value={seedInput}
                      onChange={(event) => setSeedInput(event.target.value)}
                      className="w-full rounded-[20px] border border-slate-200 bg-white px-4 py-3 text-slate-900 shadow-sm outline-none transition focus:border-slate-400"
                    />
                  </label>
                </div>
              </div>

              <div className="space-y-4 rounded-[32px] border border-slate-200 bg-slate-50/90 p-6">
                <label className="flex items-start gap-3 rounded-[22px] border border-white bg-white px-4 py-4">
                  <input
                    type="checkbox"
                    checked={experimenterMode}
                    onChange={(event) => setExperimenterMode(event.target.checked)}
                    className="mt-1 h-4 w-4 rounded border-slate-300"
                  />
                  <span className="text-sm leading-6 text-slate-600">
                    <span className="block font-semibold text-slate-900">Experimenter mode</span>
                    Show anonymous ID, waveform slot, grid position, and seed metadata during the session.
                  </span>
                </label>

                <div className="rounded-[22px] border border-white bg-white px-4 py-4 text-sm leading-6 text-slate-600">
                  <p>Flow order is fixed: Success, Error, Notification, Loading.</p>
                  <p>Anonymous IDs are global: 1-15, 16-30, 31-45, 46-60.</p>
                </div>

                <button
                  type="button"
                  onClick={handleStartSession}
                  className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800"
                >
                  Start study
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </section>
        ) : null}

        {screen === "flow-block" && currentFlow && currentStimuli ? (
          <section className="space-y-6">
            <div className="rounded-[34px] border border-white/70 bg-white/85 p-6 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                    Flow block {currentFlowIndex + 1} / {FLOW_ORDER.length}
                  </p>
                  <h2 className="mt-3 text-3xl font-bold text-slate-950">{formatFlowLabel(currentFlow)}</h2>
                  <p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">
                    The UI flow stays fixed in place. Select one anonymous vibration from the 3x5 panel, the flow demo will
                    autoplay, and then rate that haptic on the three questions.
                  </p>
                </div>
                <div className="rounded-[22px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-700">
                  Completed in this block: {currentFlowCompletedCount}/{STIMULI_PER_FLOW}
                </div>
              </div>
            </div>

            <div className="overflow-x-auto rounded-[34px] border border-white/70 bg-white/85 p-6 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur">
              <div className="grid min-w-[1360px] gap-6 xl:grid-cols-[500px_minmax(0,1fr)]">
                <div className="space-y-5">
                  <div className="rounded-[32px] border border-slate-200 bg-slate-50/80 p-5">
                    <div className="mb-4 flex items-center justify-between">
                      <div>
                        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Fixed flow view</p>
                        <p className="mt-2 text-sm text-slate-600">
                          Size and placement remain constant for every anonymous ID in this block.
                        </p>
                      </div>
                      {currentStimulus ? (
                        <button
                          type="button"
                          onClick={() => void triggerStimulusPlayback(currentStimulus)}
                          className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-slate-300 hover:bg-slate-50"
                        >
                          <RefreshCcw className="h-4 w-4" />
                          Replay
                        </button>
                      ) : null}
                    </div>
                    <div className="flex justify-center">
                      <FlowRenderer uiFlow={currentFlow} playToken={playToken} onPlaybackComplete={handlePlaybackComplete} />
                    </div>
                  </div>

                  <div className="rounded-[32px] border border-slate-200 bg-slate-50/80 p-5">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-semibold text-slate-900">Current selection</p>
                      {isPlaybackRunning ? (
                        <span className="rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">
                          Playing
                        </span>
                      ) : null}
                    </div>

                    {currentStimulus ? (
                      <div className="mt-4 space-y-3 text-sm text-slate-600">
                        <p>
                          <span className="font-semibold text-slate-900">{currentStimulus.displayLabel}</span> is active for
                          this block.
                        </p>
                        <p>Anonymous ID used for the serial trigger: {currentStimulus.anonymousId}</p>
                        {experimenterMode ? (
                          <div className="rounded-[22px] border border-white bg-white px-4 py-4 text-sm leading-6 text-slate-600">
                            <p>Waveform slot: {currentStimulus.waveformSlot}</p>
                            <p>Grid position: {currentStimulus.gridPosition}</p>
                            <p>Lookup table slot: {ANONYMOUS_ID_TO_WAVEFORM_SLOT[currentStimulus.anonymousId - 1]}</p>
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <p className="mt-4 text-sm leading-6 text-slate-600">
                        Select one anonymous vibration card from the grid to start the autoplay demo.
                      </p>
                    )}
                  </div>
                </div>

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
                  <div className="space-y-5">
                    <div className="rounded-[32px] border border-slate-200 bg-slate-50/80 p-5">
                      <div className="mb-4 flex items-center justify-between">
                        <div>
                          <p className="text-sm font-semibold text-slate-900">Anonymous vibration panel</p>
                          <p className="mt-1 text-sm text-slate-600">Select any card to trigger the same UI flow with a different haptic ID.</p>
                        </div>
                        <span className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                          3 x 5 grid
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-3">
                        {currentStimuli.map((stimulus) => {
                          const isSelected = stimulus.anonymousId === selectedStimulusId;
                          const isComplete = Boolean(blockResultsByStimulusId[stimulus.anonymousId]);

                          return (
                            <button
                              key={stimulus.anonymousId}
                              type="button"
                              onClick={() => void triggerStimulusPlayback(stimulus)}
                              className={`rounded-[24px] border p-4 text-left transition ${
                                isSelected
                                  ? "border-slate-900 bg-slate-900 text-white shadow-lg"
                                  : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                              }`}
                            >
                              <div className="flex items-start justify-between">
                                <div>
                                  <p className={`text-xs font-semibold uppercase tracking-[0.18em] ${isSelected ? "text-slate-300" : "text-slate-400"}`}>
                                    Anonymous
                                  </p>
                                  <p className="mt-2 text-2xl font-bold">{stimulus.displayLabel}</p>
                                </div>
                                {isComplete ? (
                                  <span className={`rounded-full p-1 ${isSelected ? "bg-white/15" : "bg-emerald-100 text-emerald-700"}`}>
                                    <Check className="h-4 w-4" />
                                  </span>
                                ) : null}
                              </div>
                              <p className={`mt-3 text-sm leading-6 ${isSelected ? "text-slate-200" : "text-slate-500"}`}>
                                {isComplete ? "Rated and saved" : "Select to autoplay and rate"}
                              </p>
                              {experimenterMode ? (
                                <div className={`mt-3 text-xs ${isSelected ? "text-slate-300" : "text-slate-400"}`}>
                                  ID {stimulus.anonymousId} · Slot {stimulus.waveformSlot} · Pos {stimulus.gridPosition}
                                </div>
                              ) : null}
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="flex items-center justify-end">
                      <button
                        type="button"
                        onClick={handleContinueToOverview}
                        disabled={currentFlowCompletedCount < STIMULI_PER_FLOW}
                        className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-300"
                      >
                        Continue to {formatFlowShortLabel(currentFlow)} overview
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  <div className="space-y-5">
                    <div className="rounded-[32px] border border-slate-200 bg-slate-50/80 p-5">
                      <div className="mb-4">
                        <p className="text-sm font-semibold text-slate-900">Ratings for the current anonymous ID</p>
                        <p className="mt-1 text-sm leading-6 text-slate-600">
                          Answer after you replay or experience the selected haptic.
                        </p>
                      </div>

                      <div className="space-y-4">
                        {LIKERT_QUESTIONS.map((question) => (
                          <LikertQuestionRow
                            key={question.key}
                            disabled={!currentStimulus}
                            prompt={question.prompt}
                            selectedValue={currentRatings[question.key]}
                            onSelect={(value) => handleRatingSelect(question.key, value)}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        ) : null}

        {screen === "flow-overview" && currentFlow && currentStimuli ? (
          <section className="space-y-6">
            <div className="rounded-[34px] border border-white/70 bg-white/85 p-6 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur">
              <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                Overview step {currentFlowIndex + 1} / {FLOW_ORDER.length}
              </p>
              <h2 className="mt-3 text-3xl font-bold text-slate-950">
                {formatFlowShortLabel(currentFlow)} best and worst matches
              </h2>
              <p className="mt-3 max-w-3xl text-base leading-7 text-slate-600">
                Pick exactly three anonymous IDs as the best match for this UI event, then pick one or two as the worst
                match. These sets cannot overlap.
              </p>
            </div>

            <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
              <div className="rounded-[34px] border border-white/70 bg-white/85 p-6 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur">
                <div className="mb-5 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-[24px] border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-600">
                    <p className="font-semibold text-slate-900">Top 3 selected</p>
                    <p className="mt-2">{currentOverviewDraft.top3Ids.length}/3</p>
                  </div>
                  <div className="rounded-[24px] border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-600">
                    <p className="font-semibold text-slate-900">Bottom selected</p>
                    <p className="mt-2">{currentOverviewDraft.bottomIds.length}/2</p>
                  </div>
                  <div className="rounded-[24px] border border-slate-200 bg-slate-50 px-4 py-4 text-sm text-slate-600">
                    <p className="font-semibold text-slate-900">Block completion</p>
                    <p className="mt-2">{currentFlowCompletedCount}/15 rated</p>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                  {currentStimuli.map((stimulus) => {
                    const inTop = currentOverviewDraft.top3Ids.includes(stimulus.anonymousId);
                    const inBottom = currentOverviewDraft.bottomIds.includes(stimulus.anonymousId);

                    return (
                      <div key={stimulus.anonymousId} className="rounded-[24px] border border-slate-200 bg-slate-50 p-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Anonymous</p>
                            <p className="mt-2 text-2xl font-bold text-slate-950">{stimulus.displayLabel}</p>
                          </div>
                          <div className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-500">
                            {stimulus.anonymousId}
                          </div>
                        </div>
                        <div className="mt-4 flex gap-2">
                          <button
                            type="button"
                            onClick={() => handleToggleOverviewSelection("top3Ids", stimulus.anonymousId)}
                            className={`flex-1 rounded-full px-3 py-2 text-sm font-semibold transition ${
                              inTop
                                ? "bg-emerald-600 text-white"
                                : "border border-slate-200 bg-white text-slate-700 hover:border-slate-300"
                            }`}
                          >
                            {inTop ? "Best match" : "Mark best"}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleToggleOverviewSelection("bottomIds", stimulus.anonymousId)}
                            className={`flex-1 rounded-full px-3 py-2 text-sm font-semibold transition ${
                              inBottom
                                ? "bg-rose-600 text-white"
                                : "border border-slate-200 bg-white text-slate-700 hover:border-slate-300"
                            }`}
                          >
                            {inBottom ? "Worst match" : "Mark worst"}
                          </button>
                        </div>
                        {experimenterMode ? (
                          <p className="mt-3 text-xs text-slate-400">
                            Slot {stimulus.waveformSlot} · Grid {stimulus.gridPosition}
                          </p>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-5 rounded-[34px] border border-white/70 bg-white/85 p-6 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur">
                <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-5">
                  <p className="text-sm font-semibold text-slate-900">Why did your best choices match?</p>
                  <textarea
                    value={currentOverviewDraft.topReason}
                    onChange={(event) => handleOverviewTextChange("topReason", event.target.value)}
                    rows={5}
                    placeholder="Optional open response"
                    className="mt-4 w-full rounded-[20px] border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-slate-400"
                  />
                </div>
                <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-5">
                  <p className="text-sm font-semibold text-slate-900">Why did your worst choices not match?</p>
                  <textarea
                    value={currentOverviewDraft.bottomReason}
                    onChange={(event) => handleOverviewTextChange("bottomReason", event.target.value)}
                    rows={5}
                    placeholder="Optional open response"
                    className="mt-4 w-full rounded-[20px] border border-slate-200 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-slate-400"
                  />
                </div>

                <button
                  type="button"
                  onClick={handleSaveOverview}
                  className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800"
                >
                  {currentFlowIndex === FLOW_ORDER.length - 1 ? "Finish study" : `Continue to ${formatFlowShortLabel(FLOW_ORDER[currentFlowIndex + 1])}`}
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          </section>
        ) : null}

        {screen === "completion" ? (
          <section className="mx-auto max-w-5xl rounded-[40px] border border-white/70 bg-white/85 p-8 shadow-[0_30px_120px_rgba(15,23,42,0.12)] backdrop-blur sm:p-10 lg:p-12">
            <div className="grid gap-8 lg:grid-cols-[1fr_0.9fr]">
              <div className="space-y-6">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Study complete</p>
                  <h2 className="mt-3 text-4xl font-bold text-slate-950">Session ready to export</h2>
                  <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
                    All four flow blocks and overview selections are complete. Export the detailed ratings CSV and the
                    overview CSV, then optionally open the follow-up Google Form.
                  </p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-5">
                    <p className="text-sm font-semibold text-slate-900">Participant</p>
                    <p className="mt-2 text-lg font-bold text-slate-950">{participantId}</p>
                  </div>
                  <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-5">
                    <p className="text-sm font-semibold text-slate-900">Anonymous IDs completed</p>
                    <p className="mt-2 text-lg font-bold text-slate-950">{overallCompletedCount} / {TOTAL_ANON_STIMULI}</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={handleDownloadBlockResults}
                    className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:bg-slate-800"
                  >
                    <Download className="h-4 w-4" />
                    Export ratings CSV
                  </button>
                  <button
                    type="button"
                    onClick={handleDownloadOverviewResults}
                    className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
                  >
                    <Download className="h-4 w-4" />
                    Export overview CSV
                  </button>
                  {googleFormsUrl ? (
                    <button
                      type="button"
                      onClick={() => window.open(googleFormsUrl, "_blank", "noopener,noreferrer")}
                      className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-6 py-3 text-sm font-semibold text-sky-700 transition hover:border-sky-300 hover:bg-sky-100"
                    >
                      <ExternalLink className="h-4 w-4" />
                      Open Google Form
                    </button>
                  ) : null}
                </div>
              </div>

              <div className="space-y-4 rounded-[32px] border border-slate-200 bg-slate-50/90 p-6">
                {FLOW_ORDER.map((flow) => {
                  const result = overviewResults[flow];
                  return (
                    <div key={flow} className="rounded-[24px] border border-white bg-white p-5 shadow-sm">
                      <p className="text-sm font-semibold text-slate-900">{formatFlowShortLabel(flow)}</p>
                      {result ? (
                        <div className="mt-3 space-y-2 text-sm leading-6 text-slate-600">
                          <p>Top 3: {result.top3Ids.map((value) => `A${String(value).padStart(2, "0")}`).join(", ")}</p>
                          <p>Bottom: {result.bottomIds.map((value) => `A${String(value).padStart(2, "0")}`).join(", ")}</p>
                        </div>
                      ) : (
                        <p className="mt-3 text-sm leading-6 text-slate-500">No overview captured.</p>
                      )}
                    </div>
                  );
                })}

                <button
                  type="button"
                  onClick={handleReset}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-6 py-3 text-sm font-semibold text-slate-700 transition hover:border-slate-300 hover:bg-slate-50"
                >
                  <RotateCcw className="h-4 w-4" />
                  Start a new participant
                </button>
              </div>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
