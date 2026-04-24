import { useEffect, useState } from "react";
import { ArrowDown, CheckCircle2, Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { FlowShell } from "./FlowShell";

interface LoadingFlowProps {
  playToken?: number;
  onPlaybackComplete?: () => void;
}

const ITEMS = [
  { id: 1, title: "Morning standup", time: "9:00 AM" },
  { id: 2, title: "Design review", time: "11:30 AM" },
  { id: 3, title: "Lunch break", time: "12:30 PM" },
  { id: 4, title: "Code review", time: "2:00 PM" },
];

type RefreshPhase = "idle" | "pulling" | "loading" | "complete";

export function LoadingFlow({ playToken = 0, onPlaybackComplete }: LoadingFlowProps) {
  const [phase, setPhase] = useState<RefreshPhase>("idle");

  useEffect(() => {
    setPhase("idle");

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setPhase("pulling"), 260),
      window.setTimeout(() => setPhase("loading"), 920),
      window.setTimeout(() => setPhase("complete"), 1740),
      window.setTimeout(() => onPlaybackComplete?.(), 2680),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  const isPulling = phase === "pulling";
  const isLoading = phase === "loading";
  const showSuccess = phase === "complete";
  const contentOffset = isPulling ? 72 : isLoading ? 38 : 0;

  return (
    <FlowShell
      accentClassName="from-indigo-50 via-white to-violet-100/75"
      label="Loading"
      title="Pull to Refresh"
      subtitle="A downward pull, loading spinner, and clean refreshed confirmation."
    >
      <div className="relative flex flex-1 flex-col overflow-hidden">
        <div className="absolute inset-x-0 top-0 z-20 flex justify-center">
          <motion.div
            initial={false}
            animate={{ opacity: phase === "idle" ? 0 : 1, y: phase === "idle" ? -26 : 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 24 }}
            className={`flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold shadow-lg ${
              showSuccess ? "bg-emerald-600 text-white" : "bg-white text-violet-700"
            }`}
          >
            {isPulling ? <ArrowDown className="h-4 w-4" /> : null}
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {showSuccess ? <CheckCircle2 className="h-4 w-4" /> : null}
            {isPulling ? "Pull to refresh" : isLoading ? "Refreshing..." : showSuccess ? "Refreshed" : ""}
          </motion.div>
        </div>

        <motion.div
          animate={{ y: contentOffset }}
          transition={{ type: "spring", stiffness: 230, damping: 24 }}
          className="flex flex-1 flex-col"
        >
          <div className="mb-6 rounded-[24px] border border-violet-100 bg-violet-50/85 p-4">
            <p className="text-xs font-medium uppercase tracking-[0.2em] text-violet-500">Sync state</p>
            <div className="mt-3 flex items-center justify-between gap-4">
              <p className="text-sm leading-6 text-slate-600">
                {showSuccess ? "Your schedule is up to date." : "Refreshing your schedule from the latest feed."}
              </p>
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm">
                {showSuccess ? (
                  <CheckCircle2 className="h-5 w-5 text-emerald-600" />
                ) : (
                  <Loader2 className={`h-5 w-5 text-violet-600 ${isLoading ? "animate-spin" : ""}`} />
                )}
              </div>
            </div>
          </div>

          <div className="space-y-3">
            {ITEMS.map((item) => (
              <motion.div
                key={item.id}
                animate={{ opacity: isLoading ? 0.7 : 1, scale: isLoading ? 0.985 : 1 }}
                className="rounded-[22px] border border-slate-100 bg-white/95 p-4 shadow-sm"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-slate-900">{item.title}</h4>
                    <p className="mt-1 text-xs text-slate-400">{item.time}</p>
                  </div>
                  <div className={`h-2.5 w-2.5 rounded-full ${showSuccess ? "bg-emerald-500" : "bg-violet-500"}`} />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <AnimatePresence>
          {showSuccess ? (
            <motion.div
              initial={{ opacity: 0, y: -18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              transition={{ type: "spring", stiffness: 280, damping: 22 }}
              className="absolute left-1/2 top-2 z-30 -translate-x-1/2"
            >
              <div className="flex items-center gap-2 rounded-full bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg">
                <CheckCircle2 className="h-4 w-4" />
                Refreshed
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </FlowShell>
  );
}
