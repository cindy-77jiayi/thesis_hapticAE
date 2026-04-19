import type { ReactNode } from "react";

interface FlowShellProps {
  accentClassName: string;
  title: string;
  subtitle: string;
  label: string;
  children: ReactNode;
}

export function FlowShell({ accentClassName, title, subtitle, label, children }: FlowShellProps) {
  return (
    <div
      className={`relative mx-auto flex h-[560px] w-full max-w-[360px] flex-col overflow-hidden rounded-[34px] border border-white/70 bg-gradient-to-b ${accentClassName} shadow-[0_30px_90px_rgba(15,23,42,0.22)]`}
    >
      <div className="absolute left-1/2 top-3 h-1.5 w-20 -translate-x-1/2 rounded-full bg-slate-900/90" />
      <div className="flex flex-1 flex-col px-6 pb-6 pt-10">
        <div className="mb-6">
          <div className="mb-3 inline-flex rounded-full border border-white/70 bg-white/85 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500 shadow-sm backdrop-blur">
            {label}
          </div>
          <h3 className="text-2xl font-bold text-slate-950">{title}</h3>
          <p className="mt-2 text-sm leading-6 text-slate-500">{subtitle}</p>
        </div>
        <div className="flex flex-1 flex-col">{children}</div>
      </div>
    </div>
  );
}
