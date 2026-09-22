import type { AlertsResponse, Bootstrap, Route, Mode, SttResponse, TtsResponse, DescribeResponse } from '@/types';
// Same origin: Kaggle sets an HttpOnly cookie from the token on the initial page.
const token = new URLSearchParams(location.search).get('token');
export function apiUrl(path: string): string {
  const url = new URL(path, location.origin);
  if (url.origin !== location.origin) throw new Error('API phải cùng máy chủ.');
  // Also works behind the local Vite proxy before a session cookie is established.
  if (token) url.searchParams.set('token', token);
  return url.href;
}
async function request<T>(path: string, init: RequestInit = {}, timeout = 30000): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  const timer = setTimeout(abort, timeout);
  init.signal?.addEventListener('abort', abort, { once: true });
  if (init.signal?.aborted) abort();
  try {
    const res = await fetch(apiUrl(path), { ...init, credentials: 'same-origin', signal: controller.signal });
    if (!res.ok) throw new Error(res.status === 401
      ? 'Phiên truy cập đã hết hạn. Hãy mở lại liên kết Kaggle mới.'
      : 'Máy chủ chưa trả lời được. Vui lòng thử lại.');
    return await res.json() as T;
  } finally { clearTimeout(timer); init.signal?.removeEventListener('abort', abort); }
}
const json = (body: unknown, signal?: AbortSignal): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal,
});
export const bootstrap = (signal?: AbortSignal) => request<Bootstrap>('/v1/bootstrap', {signal});
export const fetchAlerts = (signal?: AbortSignal) => request<AlertsResponse>('/v1/alerts', {signal});
export const intent = (text: string, current_mode: Mode, resume_mode: Mode, signal?: AbortSignal) =>
  request<Route>('/v1/intent', json({text, current_mode, resume_mode}, signal));
export function stt(audio: Blob, currentMode: Mode, resumeMode: Mode, turnId: string, signal?: AbortSignal) {
  const form = new FormData();
  form.append('audio', audio, 'utterance.wav'); form.append('current_mode', currentMode);
  form.append('resume_mode', resumeMode); form.append('turn_id', turnId);
  return request<SttResponse>('/v1/stt', {method:'POST', body:form, signal});
}
export const tts = (text: string, signal?: AbortSignal) => request<TtsResponse>('/v1/tts', json({text}, signal));
export const describe = (image: string, question: string, requestId: string, signal?: AbortSignal) =>
  request<DescribeResponse>('/v1/describe', json({image_base64:image, question, request_id:requestId, tts:true}, signal), 60000);
export function buildWsUrl() {
  const url = new URL(apiUrl('/v1/vision'));
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.href;
}
export const buildAudioUrl = (code: string, pattern = '/audio/{code}.wav') => apiUrl(pattern.replace('{code}', encodeURIComponent(code)));

export async function audioBytes(path:string,signal?:AbortSignal) {
  const controller=new AbortController(),abort=()=>controller.abort();
  const timer=setTimeout(abort,15000);
  signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
  try {
    const res=await fetch(apiUrl(path),{signal:controller.signal,credentials:'same-origin'});
    if(!res.ok)throw new Error('Không tải được âm thanh. Hãy thử kết nối lại.');
    return await res.arrayBuffer();
  }finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
