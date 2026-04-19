import { useEffect, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { motion } from "motion/react";

import { FlowShell } from "./FlowShell";

interface SuccessFlowProps {
  playToken?: number;
  onPlaybackComplete?: () => void;
}

const SWIPE_DISTANCE = 216;

export function SuccessFlow({ playToken = 0, onPlaybackComplete }: SuccessFlowProps) {
  const [isSwiping, setIsSwiping] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    setIsSwiping(false);
    setIsComplete(false);

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setIsSwiping(true), 260),
      window.setTimeout(() => setIsComplete(true), 1480),
      window.setTimeout(() => onPlaybackComplete?.(), 2450),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  return (
    <FlowShell
      accentClassName="from-emerald-50 via-white to-emerald-100/70"
      label="Success"
      title="Payment Confirmation"
      subtitle="A confident success cue for a completed payment."
    >
      <div className="space-y-4">
        <div className="rounded-[24px] border border-emerald-100 bg-white/90 p-5 shadow-sm">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-500">Amount</span>
              <span className="text-lg font-semibold text-slate-950">$124.50</span>
            </div>
            <div className="h-px bg-slate-100" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-500">Recipient</span>
              <span className="text-sm font-medium text-slate-900">Metro Transit</span>
            </div>
            <div className="h-px bg-slate-100" />
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-500">Reference</span>
              <span className="text-sm font-medium text-slate-900">#PAY-2048</span>
            </div>
          </div>
        </div>

        {!isComplete ? (
          <div className="pt-3">
            <div className="mb-3 flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
              <span>Swipe to confirm</span>
              <span className={isSwiping ? "text-emerald-600" : ""}>Ready</span>
            </div>
            <div className="relative h-16 overflow-hidden rounded-[22px] bg-slate-100">
              <motion.div
                animate={{ opacity: isSwiping ? 1 : 0.35 }}
                className="absolute inset-0 bg-[linear-gradient(90deg,rgba(16,185,129,0.18)_0%,rgba(16,185,129,0.03)_100%)]"
              />
              <motion.div
                animate={{ x: isSwiping ? SWIPE_DISTANCE : 0 }}
                transition={{ duration: 0.95, ease: "easeInOut" }}
                className="absolute left-2 top-2 flex h-12 w-12 items-center justify-center rounded-[18px] bg-emerald-500 shadow-lg"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="text-white">
                  <path
                    d="M9 18l6-6-6-6"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </motion.div>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-medium text-slate-400">Slide to confirm payment</span>
              </div>
            </div>
          </div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.94, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            className="rounded-[28px] border border-emerald-200 bg-white/95 p-7 text-center shadow-sm"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 260, damping: 18 }}
              className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-emerald-100"
            >
              <CheckCircle2 className="h-10 w-10 text-emerald-600" />
            </motion.div>
            <h4 className="text-xl font-bold text-slate-950">Payment Successful</h4>
            <p className="mt-2 text-sm leading-6 text-slate-500">
              Transaction approved and logged for review.
            </p>
            <div className="mt-6 space-y-2 rounded-[18px] bg-emerald-50/80 p-4 text-left text-sm">
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Transaction ID</span>
                <span className="font-medium text-slate-900">TXN-98234</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Status</span>
                <span className="font-medium text-emerald-700">Settled</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </FlowShell>
  );
}
