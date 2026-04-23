import { useEffect, useState } from "react";
import { AlertCircle, Check, CreditCard, Loader2, Mail, User, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { FlowShell } from "./FlowShell";

interface ErrorFlowProps {
  playToken?: number;
  onPlaybackComplete?: () => void;
}

type ErrorPhase = "idle" | "checked" | "submitting" | "error";

export function ErrorFlow({ playToken = 0, onPlaybackComplete }: ErrorFlowProps) {
  const [phase, setPhase] = useState<ErrorPhase>("idle");

  useEffect(() => {
    setPhase("idle");

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setPhase("checked"), 300),
      window.setTimeout(() => setPhase("submitting"), 880),
      window.setTimeout(() => setPhase("error"), 1500),
      window.setTimeout(() => onPlaybackComplete?.(), 3060),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  const isChecked = phase === "checked" || phase === "submitting" || phase === "error";
  const isSubmitting = phase === "submitting";
  const showError = phase === "error";

  return (
    <FlowShell
      accentClassName="from-rose-50 via-white to-amber-50"
      label="Error"
      title="Failed Submission"
      subtitle="A visible destructive selection, pressed button, and blocked result."
    >
      <div className="relative flex flex-1 flex-col">
        <motion.div animate={showError ? { x: [0, -8, 8, -5, 5, 0] } : { x: 0 }} className="space-y-3">
          <div className="rounded-[24px] border border-slate-100 bg-white/95 p-5 shadow-sm">
            <div className="mb-4 flex items-center gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100">
                <User className="h-5 w-5 text-slate-500" />
              </div>
              <div>
                <p className="text-xs text-slate-400">Full name</p>
                <p className="text-sm font-semibold text-slate-900">Jordan Lee</p>
              </div>
            </div>
            <div className="mb-4 h-px bg-slate-100" />
            <div className="flex items-center gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-slate-100">
                <Mail className="h-5 w-5 text-slate-500" />
              </div>
              <div>
                <p className="text-xs text-slate-400">Email</p>
                <p className="text-sm font-semibold text-slate-900">jordan.lee@email.com</p>
              </div>
            </div>
          </div>

          <div className="rounded-[24px] border border-slate-100 bg-white/95 p-5 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-sky-100">
                <CreditCard className="h-5 w-5 text-sky-600" />
              </div>
              <div>
                <p className="text-xs text-slate-400">Current plan</p>
                <p className="text-sm font-semibold text-slate-900">Premium Plan</p>
                <p className="mt-1 text-xs text-sky-600">Renews May 17, 2026</p>
              </div>
            </div>
          </div>

          <motion.div
            animate={{
              borderColor: isChecked ? "rgba(239,68,68,0.72)" : "rgba(226,232,240,0.9)",
              backgroundColor: isChecked ? "rgba(254,226,226,0.98)" : "rgba(255,255,255,0.95)",
              scale: isChecked ? 1.015 : 1,
            }}
            transition={{ type: "spring", stiffness: 260, damping: 20 }}
            className="rounded-[24px] border p-5 shadow-sm"
          >
            <div className="flex items-start gap-4">
              <motion.div
                animate={{ scale: isChecked ? [1, 1.22, 1] : 1 }}
                className={`mt-0.5 flex h-6 w-6 items-center justify-center rounded-md border ${
                  isChecked ? "border-rose-600 bg-rose-600" : "border-slate-300 bg-white"
                }`}
              >
                {isChecked ? <Check className="h-4 w-4 text-white" /> : null}
              </motion.div>
              <div>
                <p className="text-sm font-bold text-slate-900">Delete my account</p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  Permanently remove the account and all stored profile data.
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>

        <div className="mt-auto pt-5">
          <motion.button
            type="button"
            animate={{
              scale: isSubmitting ? 0.94 : 1,
              backgroundColor: showError ? "rgb(225,29,72)" : isSubmitting ? "rgb(244,63,94)" : "rgb(15,23,42)",
              boxShadow: isSubmitting || showError ? "0 16px 36px rgba(225,29,72,0.32)" : "0 10px 24px rgba(15,23,42,0.16)",
            }}
            transition={{ type: "spring", stiffness: 320, damping: 22 }}
            className="flex w-full items-center justify-center gap-3 rounded-[22px] py-4 text-sm font-bold text-white shadow-sm"
          >
            {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {showError ? <AlertCircle className="h-4 w-4" /> : null}
            {showError ? "Submission failed" : isSubmitting ? "Saving changes..." : "Save Changes"}
          </motion.button>
        </div>

        <AnimatePresence>
          {showError ? (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 rounded-[30px] bg-slate-950/22 backdrop-blur-[2px]"
              />
              <motion.div
                initial={{ opacity: 0, y: 30, scale: 0.94 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 12, scale: 0.96 }}
                transition={{ type: "spring", stiffness: 260, damping: 22 }}
                className="absolute inset-x-0 bottom-3 z-10"
              >
                <div className="rounded-[28px] border border-rose-100 bg-white/98 p-6 shadow-2xl">
                  <div className="mb-4 flex items-start justify-between">
                    <motion.div
                      initial={{ scale: 0.55, rotate: -20 }}
                      animate={{ scale: 1, rotate: 0 }}
                      className="flex h-12 w-12 items-center justify-center rounded-full bg-rose-600 text-white"
                    >
                      <AlertCircle className="h-6 w-6" />
                    </motion.div>
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-100">
                      <X className="h-4 w-4 text-slate-500" />
                    </div>
                  </div>
                  <h4 className="text-xl font-bold text-slate-950">Action Cannot Be Completed</h4>
                  <p className="mt-2 text-sm leading-6 text-slate-500">
                    Active subscriptions must be canceled before account deletion.
                  </p>
                  <div className="mt-5 rounded-[18px] bg-rose-50 px-4 py-3 text-xs font-bold text-rose-700">
                    Please contact support for the next step.
                  </div>
                </div>
              </motion.div>
            </>
          ) : null}
        </AnimatePresence>
      </div>
    </FlowShell>
  );
}
