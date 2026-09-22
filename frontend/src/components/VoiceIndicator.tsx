import type { AppStatus, AppMode } from '@/types';

interface Props {
  status: AppStatus;
  mode: AppMode;
  transcript: string;
}

const STATUS_LABELS: Record<AppStatus, string> = {
  listening:   'Đang nghe',
  speech:      'Đã nhận giọng nói',
  transcribing:'Đang nhận dạng…',
  waiting:     'Đang kiểm tra cảnh',
  speaking:    'Trợ lý đang nói',
  paused:      'Tạm dừng · vẫn nghe lệnh',
  error:       'Đã xảy ra lỗi',
  lost:        'Mất kết nối',
};

const MODE_LABELS: Record<NonNullable<AppMode>, string> = {
  guide:    'Dẫn đường',
  narration:'Thuyết minh',
};

function getIndicatorStyle(status: AppStatus, mode: AppMode): { ring: string; dot: string; animate: boolean } {
  if (status === 'error' || status === 'lost') return { ring: '#ef4444', dot: '#ef4444', animate: false };
  if (status === 'paused') return { ring: '#7a98c4', dot: '#7a98c4', animate: false };
  if (status === 'speaking') return { ring: '#10b981', dot: '#10b981', animate: true };
  if (status === 'speech' || status === 'transcribing' || status === 'waiting') return { ring: '#f59e0b', dot: '#f59e0b', animate: true };
  if (mode === 'guide') return { ring: '#10b981', dot: '#10b981', animate: true };
  if (mode === 'narration') return { ring: '#f59e0b', dot: '#f59e0b', animate: true };
  return { ring: '#10b981', dot: '#10b981', animate: true };
}

function MicIcon({ color }: { color: string }) {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" fill={color} />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <path d="M12 19v4M8 23h8" stroke={color} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function SpeakingIcon({ color }: { color: string }) {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M11 5L6 9H2v6h4l5 4V5z" fill={color} />
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07" stroke={color} strokeWidth="2" strokeLinecap="round" />
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14" stroke={color} strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function PauseIcon({ color }: { color: string }) {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="6" y="4" width="4" height="16" rx="1" fill={color} />
      <rect x="14" y="4" width="4" height="16" rx="1" fill={color} />
    </svg>
  );
}

export default function VoiceIndicator({ status, mode, transcript }: Props) {
  const { ring, dot, animate } = getIndicatorStyle(status, mode);
  const isPaused = status === 'paused';
  const isSpeaking = status === 'speaking';

  return (
    <div className="flex flex-col items-center gap-4" aria-live="polite" aria-atomic="true">
      {/* animated ring + dot */}
      <div className="relative flex items-center justify-center" style={{ width: 120, height: 120 }}>
        {animate && (
          <>
            <div
              className="absolute inset-0 rounded-full anim-pulse-ring"
              style={{ border: `2px solid ${ring}`, borderRadius: '50%' }}
            />
            <div
              className="absolute inset-0 rounded-full anim-pulse-ring-slow"
              style={{ border: `1px solid ${ring}`, borderRadius: '50%' }}
            />
          </>
        )}
        <div
          className={`w-20 h-20 rounded-full flex items-center justify-center ${animate && !isSpeaking ? 'anim-breathe' : ''}`}
          style={{
            background: `${dot}18`,
            border: `2px solid ${dot}50`,
            boxShadow: animate ? `0 0 24px ${dot}30` : 'none',
          }}
          aria-hidden="true"
        >
          {isPaused ? (
            <PauseIcon color={dot} />
          ) : isSpeaking ? (
            <SpeakingIcon color={dot} />
          ) : (
            <MicIcon color={dot} />
          )}
        </div>
      </div>

      {/* mode badge */}
      {mode && (
        <div
          className="px-3 py-1 rounded-full text-xs font-semibold tracking-wide uppercase"
          style={{
            background: mode === 'guide' ? '#10b98120' : '#f59e0b20',
            color: mode === 'guide' ? '#10b981' : '#f59e0b',
            border: `1px solid ${mode === 'guide' ? '#10b98140' : '#f59e0b40'}`,
          }}
          aria-label={`Chế độ: ${MODE_LABELS[mode]}`}
        >
          {MODE_LABELS[mode]}
        </div>
      )}

      {/* status text */}
      <p
        className="text-xl font-semibold tracking-tight"
        style={{ color: 'var(--text)' }}
        aria-label={`Trạng thái: ${STATUS_LABELS[status]}`}
      >
        {STATUS_LABELS[status]}
      </p>

      {/* transcript */}
      {transcript && (
        <p
          className="text-base text-center max-w-sm leading-relaxed anim-fade-up px-4"
          style={{ color: 'var(--text-muted)' }}
          aria-label={`Câu nói nhận được: ${transcript}`}
        >
          "{transcript}"
        </p>
      )}
    </div>
  );
}
