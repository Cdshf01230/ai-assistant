interface Props {
  onStart: () => void;
  initializing: boolean;
}

export default function PermissionScreen({ onStart, initializing }: Props) {
  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-6 text-center"
      style={{ background: 'var(--bg)' }}
    >
      {/* logo mark */}
      <div className="mb-10">
        <div
          className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-4"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
          aria-hidden="true"
        >
          <svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
            <circle cx="20" cy="20" r="16" stroke="#10b981" strokeWidth="2" />
            <circle cx="20" cy="20" r="8" fill="#10b981" opacity="0.2" />
            <path d="M20 8 L20 14 M20 26 L20 32 M8 20 L14 20 M26 20 L32 20" stroke="#10b981" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--text)' }}>
          Trợ lý thị giác
        </h1>
        <p className="mt-2 text-base font-light" style={{ color: 'var(--text-muted)' }}>
          Hỗ trợ định hướng và mô tả môi trường
        </p>
      </div>

      {/* permission notice */}
      <div
        className="w-full max-w-sm rounded-xl px-5 py-4 mb-8 text-sm"
        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
        role="note"
        aria-label="Yêu cầu quyền truy cập"
      >
        <div className="flex gap-3 items-start">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" className="flex-shrink-0 mt-0.5" aria-hidden="true">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" fill="#f59e0b" />
          </svg>
          <p style={{ color: 'var(--text-muted)' }}>
            Ứng dụng cần quyền sử dụng <strong style={{ color: 'var(--text)' }}>camera</strong> và <strong style={{ color: 'var(--text)' }}>micro</strong> để hoạt động.
          </p>
        </div>
      </div>

      {/* start button */}
      <button
        onClick={onStart}
        disabled={initializing}
        className="w-full max-w-sm rounded-2xl font-semibold text-xl transition-all active:scale-95 disabled:opacity-60 disabled:cursor-not-allowed"
        style={{
          background: initializing ? 'var(--surface)' : '#10b981',
            color: initializing ? 'var(--text-muted)' : '#042c24',
          border: 'none',
          padding: '20px 0',
          minHeight: '64px',
        }}
        aria-label="Bắt đầu trợ lý thị giác"
        aria-busy={initializing}
      >
        {initializing ? (
          <span className="flex items-center justify-center gap-3">
            <svg className="anim-spin" width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31 10" />
            </svg>
            Đang khởi động…
          </span>
        ) : (
          'Bắt đầu trợ lý'
        )}
      </button>

      <p className="mt-6 text-xs" style={{ color: 'var(--text-muted)' }}>
        Ứng dụng hỗ trợ người khiếm thị và thị lực kém
      </p>
    </div>
  );
}
