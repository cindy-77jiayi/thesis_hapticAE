import { useEffect, useState } from "react";
import { Bell, CheckCircle2, Send, Sparkles } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

import { FlowShell } from "./FlowShell";

interface MessageFlowProps {
  playToken?: number;
  onPlaybackComplete?: () => void;
}

interface NotificationState {
  id: number;
  message: string;
  time: string;
}

const NOTIFICATION: NotificationState = {
  id: 1,
  message: "New message from Sarah: The mockups are ready for review.",
  time: "Just now",
};

type NotificationPhase = "idle" | "pressing" | "sent" | "received";

export function MessageFlow({ playToken = 0, onPlaybackComplete }: MessageFlowProps) {
  const [notifications, setNotifications] = useState<NotificationState[]>([]);
  const [phase, setPhase] = useState<NotificationPhase>("idle");

  useEffect(() => {
    setNotifications([]);
    setPhase("idle");

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setPhase("pressing"), 260),
      window.setTimeout(() => setPhase("sent"), 640),
      window.setTimeout(() => {
        setNotifications([NOTIFICATION]);
        setPhase("received");
      }, 1040),
      window.setTimeout(() => onPlaybackComplete?.(), 2920),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  const isPressing = phase === "pressing";
  const isSent = phase === "sent";
  const isReceived = phase === "received";

  return (
    <FlowShell
      accentClassName="from-sky-50 via-white to-cyan-100/70"
      label="Notification"
      title="Notification Received"
      subtitle="A clear send action followed by a visible incoming toast."
    >
      <div className="relative flex flex-1 flex-col">
        <div className="space-y-3">
          <motion.div
            animate={{
              borderColor: isReceived ? "rgba(14,165,233,0.38)" : "rgba(241,245,249,1)",
              backgroundColor: isReceived ? "rgba(240,249,255,0.95)" : "rgba(255,255,255,0.95)",
            }}
            className="rounded-[24px] border p-5 shadow-sm"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <motion.div
                  animate={{ scale: isReceived ? [1, 1.35, 1] : 1 }}
                  className="h-2.5 w-2.5 rounded-full bg-sky-500 shadow-[0_0_0_6px_rgba(14,165,233,0.14)]"
                />
                <span className="text-sm font-medium text-slate-900">Push notifications</span>
              </div>
              <span className="rounded-full bg-sky-50 px-2 py-1 text-[11px] font-semibold text-sky-700">Enabled</span>
            </div>
          </motion.div>

          <div className="rounded-[24px] border border-slate-100 bg-white/95 p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Bell className="h-4 w-4 text-slate-400" />
                <span className="text-sm font-medium text-slate-900">Active alerts</span>
              </div>
              <motion.span
                key={notifications.length}
                initial={{ scale: 1.7, backgroundColor: "rgb(2,132,199)", color: "rgb(255,255,255)" }}
                animate={{ scale: 1, backgroundColor: "rgb(240,249,255)", color: "rgb(3,105,161)" }}
                className="rounded-full px-2 py-1 text-[11px] font-bold"
              >
                {notifications.length}
              </motion.span>
            </div>
          </div>

          <div className="rounded-[24px] border border-slate-100 bg-white/95 p-5 shadow-sm">
            <div className="flex items-center gap-3">
              <Sparkles className="h-4 w-4 text-cyan-500" />
              <p className="text-sm leading-6 text-slate-500">
                Incoming notices appear as lightweight toast messages without breaking context.
              </p>
            </div>
          </div>
        </div>

        <div className="mt-auto pt-5">
          <motion.button
            type="button"
            animate={{
              scale: isPressing ? 0.94 : 1,
              backgroundColor: isReceived ? "rgb(8,145,178)" : isPressing || isSent ? "rgb(14,165,233)" : "rgb(15,23,42)",
              boxShadow: isPressing || isSent || isReceived ? "0 16px 36px rgba(14,165,233,0.34)" : "0 10px 24px rgba(15,23,42,0.16)",
            }}
            transition={{ type: "spring", stiffness: 320, damping: 22 }}
            className="flex w-full items-center justify-center gap-3 rounded-[22px] py-4 text-sm font-bold text-white shadow-sm"
          >
            {isReceived ? <CheckCircle2 className="h-4 w-4" /> : <Send className="h-4 w-4" />}
            {isReceived ? "Notification received" : isSent ? "Waiting for alert..." : isPressing ? "Button pressed" : "Send test notification"}
          </motion.button>
        </div>

        <div className="absolute inset-x-0 top-1 z-20 space-y-2">
          <AnimatePresence>
            {notifications.map((notification) => (
              <motion.div
                key={notification.id}
                initial={{ opacity: 0, y: -54, scale: 0.92 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ type: "spring", stiffness: 300, damping: 20 }}
              >
                <div className="rounded-[24px] border border-sky-100 bg-white/98 p-4 shadow-2xl ring-4 ring-sky-100/70">
                  <div className="flex items-start gap-3">
                    <motion.div
                      initial={{ rotate: -18, scale: 0.75 }}
                      animate={{ rotate: 0, scale: 1 }}
                      className="flex h-11 w-11 items-center justify-center rounded-[18px] bg-sky-600 text-white"
                    >
                      <Bell className="h-5 w-5" />
                    </motion.div>
                    <div className="min-w-0 flex-1">
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <p className="text-sm font-bold text-slate-900">Notification</p>
                        <span className="text-[11px] font-medium text-slate-400">{notification.time}</span>
                      </div>
                      <p className="text-sm leading-6 text-slate-600">{notification.message}</p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </FlowShell>
  );
}
