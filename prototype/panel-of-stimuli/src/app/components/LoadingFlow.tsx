import { useEffect, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";
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

export function LoadingFlow({ playToken = 0, onPlaybackComplete }: LoadingFlowProps) {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  useEffect(() => {
    setIsRefreshing(false);
    setShowSuccess(false);

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setIsRefreshing(true), 260),
      window.setTimeout(() => {
        setIsRefreshing(false);
        setShowSuccess(true);
      }, 1500),
      window.setTimeout(() => onPlaybackComplete?.(), 2420),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  return (
    <FlowShell
      accentClassName="from-indigo-50 via-white to-violet-100/75"
      label="Loading"
      title="Loading Complete"
      subtitle="A progress-oriented cue that resolves cleanly into success."
    >
      <div className="relative flex flex-1 flex-col">
        <div className="mb-6 rounded-[24px] border border-violet-100 bg-violet-50/85 p-4">
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-violet-500">Sync state</p>
          <div className="mt-3 flex items-center justify-between gap-4">
            <p className="text-sm leading-6 text-slate-600">Refreshing your schedule from the latest feed.</p>
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm">
              <Loader2 className={`h-5 w-5 text-violet-600 ${isRefreshing ? "animate-spin" : ""}`} />
            </div>
          </div>
        </div>

        <div className="space-y-3">
          {ITEMS.map((item) => (
            <motion.div
              key={item.id}
              animate={{ opacity: isRefreshing ? 0.7 : 1, scale: isRefreshing ? 0.985 : 1 }}
              className="rounded-[22px] border border-slate-100 bg-white/95 p-4 shadow-sm"
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium text-slate-900">{item.title}</h4>
                  <p className="mt-1 text-xs text-slate-400">{item.time}</p>
                </div>
                <div className="h-2.5 w-2.5 rounded-full bg-violet-500" />
              </div>
            </motion.div>
          ))}
        </div>

        <AnimatePresence>
          {showSuccess ? (
            <motion.div
              initial={{ opacity: 0, y: -18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              transition={{ type: "spring", stiffness: 280, damping: 22 }}
              className="absolute left-1/2 top-2 -translate-x-1/2"
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
