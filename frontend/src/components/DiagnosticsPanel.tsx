import { useState } from 'react';
import type { Diagnostics } from '@/types';

interface Props {
  data: Diagnostics;
}

export default function DiagnosticsPanel({ data }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="w-full">
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2 text-sm py-2 min-h-14 transition-opacity hover:opacity-80"
        style={{ color: 'var(--text-muted)' }}
        aria-expanded={open}
        aria-controls="diag-panel"
      >
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          className="transition-transform"
          style={{ transform: open ? 'rotate(90deg)' : 'rotate(0deg)' }}
          aria-hidden="true"
        >
          <path d="M9 18l6-6-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        Chẩn đoán kỹ thuật
      </button>

      {open && (
        <div
          id="diag-panel"
          className="rounded-xl px-4 py-3 text-xs font-mono grid grid-cols-2 gap-x-6 gap-y-2"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}
          aria-label="Thông tin kỹ thuật"
        >
          <DiagRow label="FPS" value={`${data.fps.toFixed(1)} fps`} />
          <DiagRow label="Depth" value={`${data.depthLatency} ms`} />
          <DiagRow label="VLM" value={`${data.vlmLatency} ms`} />
          <DiagRow label="Server tổng" value={`${data.serverTotal} ms`} />
          <DiagRow label="Chờ queue" value={`${data.queueWait} ms`} />
          <DiagRow label="Bỏ frame cũ" value={`${data.droppedFrames}`} />
          <DiagRow label="Khoảng trống" value={`${data.freePath.toFixed(1)} m`} />
          <div className="col-span-2 flex items-center gap-2 pt-1" style={{ borderTop: '1px solid var(--border)' }}>
            <span style={{ color: 'var(--text-muted)' }}>WebSocket</span>
            <span
              className="px-2 py-0.5 rounded text-xs font-semibold"
              style={{
                background: data.wsState === 'connected' ? '#10b98120' : '#ef444420',
                color: data.wsState === 'connected' ? '#10b981' : '#ef4444',
              }}
            >
              {data.wsState}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function DiagRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text)' }}>{value}</span>
    </div>
  );
}
