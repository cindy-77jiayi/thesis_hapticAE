import { useEffect, useState } from "react";
import { CheckCircle2, ChevronRight, Loader2 } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { FlowShell } from "./FlowShell";

interface SuccessFlowProps {
  playToken?: number;
  onPlaybackComplete?: () => void;
}

type SuccessPhase = "idle" | "pressing" | "processing" | "complete";

export function SuccessFlow({ playToken = 0, onPlaybackComplete }: SuccessFlowProps) {
  const [phase, setPhase] = useState<SuccessPhase>("idle");

  useEffect(() => {
    setPhase("idle");

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setPhase("pressing"), 260),
      window.setTimeout(() => setPhase("processing"), 760),
      window.setTimeout(() => setPhase("complete"), 1580),
      window.setTimeout(() => onPlaybackComplete?.(), 2920),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  const isPressing = phase === "pressing";
  const isProcessing = phase === "processing";
  const isComplete = phase === "complete";

  return (
    <FlowShell
      accentClassName="from-emerald-50 via-white to-emerald-100/70"
      label="Success"
      title="Payment Confirmation"
      subtitle="A clear button press that resolves into a confident payment success."
    >
      <div className="relative flex flex-1 flex-col">
        <motion.div
          animate={{
            borderColor: isComplete ? "rgba(16,185,129,0.45)" : "rgba(209,250,229,1)",
            backgroundColor: isComplete ? "rgba(236,253,245,0.96)" : "rgba(255,255,255,0.9)",
          }}
          className="rounded-[24px] border p-5 shadow-sm"
        >
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
        </motion.div>

        <div className="pt-5">
          <div className="mb-3 flex items-center justify-between text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">
            <span>Primary action</span>
            <span className={isComplete ? "text-emerald-700" : isPressing || isProcessing ? "text-emerald-600" : ""}>
              {isComplete ? "Approved" : isProcessing ? "Processing" : isPressing ? "Pressed" : "Ready"}
            </span>
          </div>

          <motion.button
            type="button"
            animate={{
              scale: isPressing ? 0.94 : 1,
              backgroundColor: isComplete ? "rgb(5,150,105)" : isPressing || isProcessing ? "rgb(16,185,129)" : "rgb(15,23,42)",
              boxShadow: isPressing
                ? "0 8px 20px rgba(16,185,129,0.28)"
                : isComplete
                  ? "0 18px 42px rgba(5,150,105,0.34)"
                  : "0 12px 28px rgba(15,23,42,0.18)",
            }}
            transition={{ type: "spring", stiffness: 320, damping: 24 }}
            className="flex h-16 w-full items-center justify-center gap-3 rounded-[24px] text-sm font-bold text-white"
          >
            {isProcessing ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {isComplete ? <CheckCircle2 className="h-5 w-5" /> : null}
            {!isProcessing && !isComplete ? <ChevronRight className="h-5 w-5" /> : null}
            {isComplete ? "Payment successful" : isProcessing ? "Confirming payment..." : isPressing ? "Button pressed" : "Confirm payment"}
          </motion.button>
        </div>

        <AnimatePresence>
          {isComplete ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.92, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, y: 12 }}
              transition={{ type: "spring", stiffness: 260, damping: 18 }}
              className="mt-5 rounded-[28px] border border-emerald-200 bg-white/95 p-7 text-center shadow-lg"
            >
              <motion.div
                initial={{ scale: 0, rotate: -18 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 280, damping: 16 }}
                className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-full bg-emerald-600 text-white"
              >
                <CheckCircle2 className="h-10 w-10" />
              </motion.div>
              <h4 className="text-xl font-bold text-slate-950">Payment Successful</h4>
              <p className="mt-2 text-sm leading-6 text-slate-500">Transaction approved and logged for review.</p>
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
          ) : null}
        </AnimatePresence>
      </div>
    </FlowShell>
  );
}
