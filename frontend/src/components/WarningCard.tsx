import type { Warning } from '@/types';

const SEVERITY: Record<string, { bg: string; border: string; icon: string; label: string }> = {
  STOP:         { bg: '#ef444418', border: '#ef444460', icon: '#ef4444', label: 'DỪNG LẠI' },
  DANGER:       { bg: '#ef444418', border: '#ef444460', icon: '#ef4444', label: 'NGUY HIỂM' },
  OBSTACLE:     { bg: '#f59e0b18', border: '#f59e0b60', icon: '#f59e0b', label: 'CHƯỚNG NGẠI VẬT' },
  SLOW:         { bg: '#f59e0b18', border: '#f59e0b60', icon: '#f59e0b', label: 'GIẢM TỐC ĐỘ' },
  WARNING:      { bg: '#f59e0b18', border: '#f59e0b60', icon: '#f59e0b', label: 'CHÚ Ý' },
};

function getConfig(code: string) {
  const upper = code.toUpperCase();
  for (const [key, cfg] of Object.entries(SEVERITY)) {
    if (upper.includes(key)) return cfg;
  }
  return { bg: '#f59e0b18', border: '#f59e0b60', icon: '#f59e0b', label: 'CHÚ Ý' };
}

function StopIcon({ color }: { color: string }) {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" fill={color} />
      <path d="M12 9v4M12 17h.01" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

interface Props {
  warning: Warning;
}

export default function WarningCard({ warning }: Props) {
  const cfg = getConfig(warning.code);

  return (
    <div
      className="w-full rounded-xl px-4 py-3 anim-warning-in"
      style={{ background: warning.code === 'STOP' ? '#641c28' : '#382b13', border: `1px solid ${cfg.border}` }}
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
      aria-label={`Cảnh báo: ${warning.text}`}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          <StopIcon color={cfg.icon} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-bold tracking-widest mb-1" style={{ color: cfg.icon }}>
            {cfg.label}
          </p>
          <p className="text-xl font-semibold leading-snug" style={{ color: 'var(--text)' }}>
            {warning.text}
          </p>
        </div>
      </div>
    </div>
  );
}
