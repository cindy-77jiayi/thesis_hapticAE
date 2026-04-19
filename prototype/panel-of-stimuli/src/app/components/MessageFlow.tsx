import { useEffect, useState } from "react";
import { Bell, Sparkles } from "lucide-react";
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

export function MessageFlow({ playToken = 0, onPlaybackComplete }: MessageFlowProps) {
  const [notifications, setNotifications] = useState<NotificationState[]>([]);

  useEffect(() => {
    setNotifications([]);

    if (!playToken) {
      return undefined;
    }

    const timers = [
      window.setTimeout(() => setNotifications([NOTIFICATION]), 420),
      window.setTimeout(() => onPlaybackComplete?.(), 2300),
    ];

    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [playToken, onPlaybackComplete]);

  return (
    <FlowShell
      accentClassName="from-sky-50 via-white to-cyan-100/70"
      label="Notification"
      title="Notification Received"
      subtitle="A quick arrival pattern for transient messages and alerts."
    >
      <div className="relative flex flex-1 flex-col">
        <div className="space-y-3">
          <div className="rounded-[24px] border border-slate-100 bg-white/95 p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="h-2.5 w-2.5 rounded-full bg-sky-500 shadow-[0_0_0_6px_rgba(14,165,233,0.14)]" />
                <span className="text-sm font-medium text-slate-900">Push notifications</span>
              </div>
              <span className="rounded-full bg-sky-50 px-2 py-1 text-[11px] font-semibold text-sky-700">
                Enabled
              </span>
            </div>
          </div>

          <div className="rounded-[24px] border border-slate-100 bg-white/95 p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Bell className="h-4 w-4 text-slate-400" />
                <span className="text-sm font-medium text-slate-900">Active alerts</span>
              </div>
              <motion.span
                key={notifications.length}
                initial={{ scale: 1.35 }}
                animate={{ scale: 1 }}
                className="rounded-full bg-sky-50 px-2 py-1 text-[11px] font-semibold text-sky-700"
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

        <div className="absolute inset-x-0 top-1 z-20 space-y-2">
          <AnimatePresence>
            {notifications.map((notification) => (
              <motion.div
                key={notification.id}
                initial={{ opacity: 0, y: -28, scale: 0.96 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ type: "spring", stiffness: 260, damping: 24 }}
              >
                <div className="rounded-[24px] border border-sky-100 bg-white/98 p-4 shadow-xl">
                  <div className="flex items-start gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-[18px] bg-sky-100">
                      <Bell className="h-5 w-5 text-sky-600" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <p className="text-sm font-semibold text-slate-900">Notification</p>
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
