import type { WsState } from '@/types';

const CONFIG: Record<WsState, { label: string; color: string; dot: string }> = {
  connected:    { label: 'Đã kết nối',     color: '#10b981', dot: '#10b981' },
  connecting:   { label: 'Đang kết nối…',  color: '#f59e0b', dot: '#f59e0b' },
  disconnected: { label: 'Mất kết nối',    color: '#ef4444', dot: '#ef4444' },
  error:        { label: 'Lỗi kết nối',    color: '#ef4444', dot: '#ef4444' },
};

export default function StatusPill({ state }: { state: WsState }) {
  const cfg = CONFIG[state];
  return (
    <div
      className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium"
      style={{
        background: `${cfg.color}18`,
        border: `1px solid ${cfg.color}40`,
        color: cfg.color,
      }}
      role="status"
      aria-label={`Trạng thái kết nối: ${cfg.label}`}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{
          background: cfg.dot,
          boxShadow: state === 'connected' ? `0 0 6px ${cfg.dot}` : 'none',
        }}
        aria-hidden="true"
      />
      {cfg.label}
    </div>
  );
}
