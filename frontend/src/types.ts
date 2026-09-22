export type Screen = 'permission' | 'initializing' | 'main' | 'denied';
export type AppMode = 'guide' | 'narration' | null;
export type Mode = 'waiting' | 'guide' | 'narration' | 'paused';
export type AppStatus = 'listening' | 'speech' | 'transcribing' | 'waiting' | 'speaking' | 'paused' | 'error' | 'lost';
export type WsState = 'disconnected' | 'connecting' | 'connected' | 'error';
export interface Route {
  intent: 'switch_mode' | 'ask_description' | 'pause' | 'resume' | 'help' | 'repeat_last' | 'unknown' | null;
  mode: Mode; resume_mode?: 'guide' | 'narration';
  reply_code: string | null; reply_text?: string | null; reply_audio_url?: string | null;
  audio_priority?: number; should_describe?: boolean; question?: string | null;
}
export interface Bootstrap {
  welcome: { reply_code: string; reply_text: string };
  system_phrases: Record<string, string>; system_audio: Record<string, string | null>;
  audio_priorities: Record<string, number>;
  voice_capture: { silence_timeout_ms: number; max_utterance_ms: number; resume_after_playback_ms: number };
}
export interface AlertsResponse { phrases: Record<string,string>; audio_url_pattern: string }
export interface SttResponse extends Route { text: string; confidence: number; repeat: boolean; turn_id: string | null; audio_base64?: string }
export interface TtsResponse { message_code: string | null; cached: boolean; audio_base64: string }
export interface DescribeResponse { request_id: string | null; text: string; audio_base64?: string; provider: string; audio_priority?: number }
export interface VisionResponse {
  ok: boolean; error?: string; latency_ms?: number; depth_latency_ms?: number; vlm_latency_ms?: number;
  frame_id?: number; dropped_frames?: number; queue_wait_ms?: number; server_total_ms?: number;
  vlm_fresh?: boolean; vlm_age_ms?: number; geometry?: { free_min?: number };
  decision?: { alert: boolean; warn?: boolean; active?: boolean; message_code?: string | null; action?: string };
}
export interface Warning { code: string; text: string; ts: number }
export interface Diagnostics {
  fps:number;depthLatency:number;vlmLatency:number;wsState:WsState;freePath:number;
  queueWait:number;serverTotal:number;droppedFrames:number;
}
