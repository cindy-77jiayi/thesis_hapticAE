import { useEffect, useState } from "react";
import { AlertCircle, CreditCard, Mail, User, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { FlowShell } from "./FlowShell";

interface ErrorFlowProps {
  playToken?: number;
  onPlaybackComplete?: () => void;
}

export function ErrorFlow({ playToken = 0, onPlaybackComplete }: ErrorFlowProps) {
  const [isChecked, setIsChecked] = useState(false);
  const [showError, setShowError] = useState(false);

  useEffect(() => {
    setIsChecked(false);
    setShowError(false);

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setIsChecked(true), 350),
      window.setTimeout(() => setShowError(true), 1180),
      window.setTimeout(() => onPlaybackComplete?.(), 2450),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  return (
    <FlowShell
      accentClassName="from-rose-50 via-white to-amber-50"
      label="Error"
      title="Failed Submission"
      subtitle="An explicit stop moment that explains the failure clearly."
    >
      <div className="relative flex flex-1 flex-col">
        <div className="space-y-3">
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
              borderColor: isChecked ? "rgba(239,68,68,0.45)" : "rgba(226,232,240,0.9)",
              backgroundColor: isChecked ? "rgba(254,242,242,0.95)" : "rgba(255,255,255,0.95)",
            }}
            className="rounded-[24px] border p-5 shadow-sm"
          >
            <div className="flex items-start gap-4">
              <div
                className={`mt-0.5 flex h-5 w-5 items-center justify-center rounded-md border ${
                  isChecked ? "border-rose-500 bg-rose-500" : "border-slate-300 bg-white"
                }`}
              >
                {isChecked ? <span className="text-xs font-bold text-white">✓</span> : null}
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900">Delete my account</p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  Permanently remove the account and all stored profile data.
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        <div className="mt-auto pt-5">
          <button className="w-full rounded-[22px] bg-slate-900 py-4 text-sm font-semibold text-white shadow-sm">
            Save Changes
          </button>
        </div>

        <AnimatePresence>
          {showError ? (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 rounded-[30px] bg-slate-950/18 backdrop-blur-[2px]"
              />
              <motion.div
                initial={{ opacity: 0, y: 24, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 12, scale: 0.96 }}
                transition={{ type: "spring", stiffness: 260, damping: 22 }}
                className="absolute inset-x-0 bottom-3 z-10"
              >
                <div className="rounded-[28px] border border-rose-100 bg-white/98 p-6 shadow-xl">
                  <div className="mb-4 flex items-start justify-between">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-rose-100">
                      <AlertCircle className="h-6 w-6 text-rose-600" />
                    </div>
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-slate-100">
                      <X className="h-4 w-4 text-slate-500" />
                    </div>
                  </div>
                  <h4 className="text-xl font-bold text-slate-950">Action Cannot Be Completed</h4>
                  <p className="mt-2 text-sm leading-6 text-slate-500">
                    Active subscriptions must be canceled before account deletion.
                  </p>
                  <div className="mt-5 rounded-[18px] bg-rose-50 px-4 py-3 text-xs font-medium text-rose-700">
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
