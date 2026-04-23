import { useEffect, useState } from "react";
import { ArrowDown, CheckCircle2, Loader2, RotateCw } from "lucide-react";
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

type RefreshPhase = "idle" | "pulling" | "refreshing" | "complete";

export function LoadingFlow({ playToken = 0, onPlaybackComplete }: LoadingFlowProps) {
  const [phase, setPhase] = useState<RefreshPhase>("idle");

  useEffect(() => {
    setPhase("idle");

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setPhase("pulling"), 220),
      window.setTimeout(() => setPhase("refreshing"), 920),
      window.setTimeout(() => setPhase("complete"), 2020),
      window.setTimeout(() => onPlaybackComplete?.(), 3180),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  const isPulling = phase === "pulling";
  const isRefreshing = phase === "refreshing";
  const isComplete = phase === "complete";
  const contentOffset = isPulling ? 92 : isRefreshing ? 54 : 0;

  return (
    <FlowShell
      accentClassName="from-indigo-50 via-white to-violet-100/75"
      label="Loading"
      title="Pull to Refresh"
      subtitle="A visible pull-down interaction that loads, resolves, and confirms refresh."
    >
      <div className="relative flex flex-1 flex-col overflow-hidden">
        <div className="absolute inset-x-0 top-0 z-20 flex justify-center">
          <motion.div
            initial={false}
            animate={{
              opacity: phase === "idle" ? 0 : 1,
              y: phase === "idle" ? -32 : 0,
              scale: isComplete ? 1.04 : 1,
            }}
            transition={{ type: "spring", stiffness: 280, damping: 24 }}
            className={`flex min-w-[220px] items-center justify-center gap-2 rounded-full px-5 py-3 text-sm font-bold shadow-xl ${
              isComplete ? "bg-emerald-600 text-white" : "bg-violet-700 text-white"
            }`}
          >
            {isPulling ? <ArrowDown className="h-4 w-4 animate-bounce" /> : null}
            {isRefreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {isComplete ? <CheckCircle2 className="h-4 w-4" /> : null}
            {isPulling ? "Release to refresh" : isRefreshing ? "Refreshing..." : isComplete ? "Schedule refreshed" : "Pull"}
          </motion.div>
        </div>

        <motion.div
          animate={{ y: contentOffset }}
          transition={{ type: "spring", stiffness: 220, damping: 24 }}
          className="flex flex-1 flex-col"
        >
          <motion.div
            animate={{
              borderColor: isRefreshing ? "rgba(124,58,237,0.45)" : "rgba(221,214,254,1)",
              backgroundColor: isRefreshing ? "rgba(245,243,255,0.98)" : "rgba(245,243,255,0.85)",
            }}
            className="mb-5 rounded-[24px] border p-4"
          >
            <p className="text-xs font-medium uppercase tracking-[0.2em] text-violet-500">Sync state</p>
            <div className="mt-3 flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-slate-900">
                  {isPulling ? "Pulling timeline down" : isRefreshing ? "Fetching latest events" : isComplete ? "Refresh complete" : "Ready to refresh"}
                </p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  {isComplete ? "Updated just now." : "The list moves down before the refresh begins."}
                </p>
              </div>
              <motion.div
                animate={{ rotate: isRefreshing ? 360 : 0, scale: isComplete ? 1.12 : 1 }}
                transition={isRefreshing ? { duration: 0.75, repeat: Infinity, ease: "linear" } : { type: "spring" }}
                className={`flex h-12 w-12 items-center justify-center rounded-full shadow-sm ${
                  isComplete ? "bg-emerald-600 text-white" : "bg-white text-violet-600"
                }`}
              >
                {isComplete ? <CheckCircle2 className="h-6 w-6" /> : <RotateCw className="h-5 w-5" />}
              </motion.div>
            </div>
          </motion.div>

          <div className="space-y-3">
            {ITEMS.map((item, index) => (
              <motion.div
                key={item.id}
                animate={{
                  opacity: isRefreshing ? 0.58 : 1,
                  scale: isRefreshing ? 0.975 : 1,
                  x: isComplete ? [0, 8, 0] : 0,
                }}
                transition={{ delay: isComplete ? index * 0.05 : 0, duration: 0.35 }}
                className="rounded-[22px] border border-slate-100 bg-white/95 p-4 shadow-sm"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-slate-900">{item.title}</h4>
                    <p className="mt-1 text-xs text-slate-400">{item.time}</p>
                  </div>
                  <div className={`h-2.5 w-2.5 rounded-full ${isComplete ? "bg-emerald-500" : "bg-violet-500"}`} />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        <AnimatePresence>
          {isComplete ? (
            <motion.div
              initial={{ opacity: 0, y: 18, scale: 0.94 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 12, scale: 0.96 }}
              transition={{ type: "spring", stiffness: 280, damping: 22 }}
              className="absolute inset-x-5 bottom-1 z-30"
            >
              <div className="flex items-center justify-center gap-2 rounded-[22px] bg-emerald-600 px-5 py-3 text-sm font-bold text-white shadow-xl">
                <CheckCircle2 className="h-4 w-4" />
                New content loaded
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </FlowShell>
  );
}
