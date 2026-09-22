#!/usr/bin/env python3
"""Indoor MVP backend API."""
import asyncio, base64, io, json, os, sys, tempfile, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

DEPLOY_ROOT = Path('/home/ubuntu/ai-assistant')
ROOT = Path(os.getenv(
    'AI_ASSISTANT_ROOT',
    str(DEPLOY_ROOT if DEPLOY_ROOT.exists() else Path(__file__).resolve().parent),
))
sys.path.insert(0, str(ROOT / 'scripts'))

from dotenv import load_dotenv

load_dotenv(ROOT / '.env')

import mvp_config as cfg
import fusion as fus

cfg.ALERT_AUDIO_DIR = str(ROOT / 'assets' / 'audio')

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
STT_CONFIDENCE_THRESHOLD = 0.60   # §5: dưới ngưỡng thì phát REPEAT_PLEASE, không đoán
MODEL_STATE: dict[str, Any] = {}
DESCRIBE_MODEL = os.getenv('DESCRIBE_MODEL', 'gemini-3.5-flash-lite')
STT_LOCK = threading.Lock()
TTS_LOCK = threading.Lock()
GEMINI_LOCK = threading.Lock()


def prepare_image(data: bytes):
    from PIL import Image
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('image exceeds 5 MiB')
    image = Image.open(io.BytesIO(data)).convert('RGB')
    if max(image.size) > cfg.FRAME_LONG_EDGE:
        scale = cfg.FRAME_LONG_EDGE / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def _load_models_worker():
    try:
        load_models()
        for code, text in cfg.SYSTEM_PHRASES.items():
            path = Path(cfg.ALERT_AUDIO_DIR) / f'{code}.wav'
            if not path.is_file():
                path.write_bytes(synthesize_wav(text))
    except Exception as exc:
        MODEL_STATE['error'] = f'{type(exc).__name__}: {exc}'


def load_models():
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModelForImageTextToText, AutoProcessor, WhisperForConditionalGeneration
    MODEL_STATE['torch'] = torch
    MODEL_STATE['vlm_processor'] = AutoProcessor.from_pretrained(cfg.VLM_MODEL)
    MODEL_STATE['vlm'] = AutoModelForImageTextToText.from_pretrained(cfg.VLM_MODEL, dtype=torch.float16, attn_implementation=cfg.ATTN_IMPL, device_map='cuda:0').eval()
    depth_model = cfg.DEPTH_MODEL
    MODEL_STATE['depth_processor'] = AutoImageProcessor.from_pretrained(depth_model)
    MODEL_STATE['depth'] = AutoModelForDepthEstimation.from_pretrained(depth_model, dtype=torch.float16).to('cuda:0').eval()
    MODEL_STATE['stt_processor'] = AutoProcessor.from_pretrained(cfg.STT_MODEL)
    MODEL_STATE['stt'] = WhisperForConditionalGeneration.from_pretrained(cfg.STT_MODEL, dtype=torch.float16).to('cuda:0').eval()
    from vieneu import Vieneu
    MODEL_STATE['tts'] = Vieneu()
    MODEL_STATE['ready'] = True


def infer_vision(image):
    import numpy as np
    import torch
    vp = MODEL_STATE['vlm_processor']; vm = MODEL_STATE['vlm']; dp = MODEL_STATE['depth_processor']; dm = MODEL_STATE['depth']
    prompt = cfg.VLM_PROMPT_PIPE
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': prompt}]}, {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': 'Frame:'}]}]
    x = vp.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(vm.device); n = x['input_ids'].shape[-1]
    t0 = time.perf_counter()
    with torch.inference_mode(): out = vm.generate(**x, max_new_tokens=cfg.VLM_MAX_NEW_TOKENS, do_sample=False)
    raw = vp.decode(out[0][n:], skip_special_tokens=True).strip(); parsed = cfg.parse_pipe4(raw)
    z = dp(images=image, return_tensors='pt').to('cuda:0'); z['pixel_values'] = z['pixel_values'].half()
    with torch.inference_mode(): dout = dm(**z)
    depth = dp.post_process_depth_estimation(dout, target_sizes=[(image.height, image.width)])[0]['predicted_depth'].float().cpu().numpy().squeeze()
    return {
        'latency_ms': round((time.perf_counter() - t0) * 1000),
        'vlm_raw': raw,
        'vlm': parsed,
        'depth': {'min': float(depth.min()), 'median': float(np.median(depth)), 'max': float(depth.max())},
        '_depth_map': depth,
    }


def get_gemini_client():
    client = MODEL_STATE.get('gemini_client')
    if client is None:
        from google import genai
        client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        MODEL_STATE['gemini_client'] = client
    return client


def extract_final_text(response) -> str:
    """Read structured answer, falling back to non-thought text parts."""
    parsed = getattr(response, 'parsed', None)
    if isinstance(parsed, dict) and isinstance(parsed.get('answer'), str):
        return parsed['answer'].strip()
    candidates = getattr(response, 'candidates', None) or []
    content = getattr(candidates[0], 'content', None) if candidates else None
    parts = getattr(content, 'parts', None) or []
    answer = [
        str(part.text).strip() for part in parts
        if getattr(part, 'text', None) and not getattr(part, 'thought', False)
    ]
    raw = '\n'.join(filter(None, answer)).strip()
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict) and isinstance(payload.get('answer'), str):
            return payload['answer'].strip()
    except (TypeError, ValueError):
        pass
    return raw


def description_is_incomplete(response, text: str) -> bool:
    """Reject token-cut or sentence-fragment Gemini answers."""
    candidates = getattr(response, 'candidates', None) or []
    reason = getattr(candidates[0], 'finish_reason', '') if candidates else ''
    reason_name = getattr(reason, 'name', str(reason)).upper()
    clean = text.strip()
    tail = clean.rstrip('.!?…').casefold().split()
    dangling = {
        'đang', 'sẽ', 'và', 'hoặc', 'nhưng', 'ở', 'có', 'là', 'một',
        'những', 'các', 'phía', 'bên', 'trên', 'dưới', 'trong', 'với',
        'để', 'của', 'khi', 'do', 'từ', 'đến', 'về',
    }
    return ('MAX_TOKENS' in reason_name or len(clean.split()) < 4
            or clean[-1:] not in '.!?…' or bool(tail and tail[-1] in dangling))


def describe_with_api(image, question):
    """Kênh thuyết minh (non-critical): Gemini thinking-LOW + giới hạn 2 câu.

    Gemini 3.5 Flash Lite là mặc định để ưu tiên độ trễ; TTS chạy sau khi đã
    nhận được câu trả lời hoàn chỉnh. Kênh cảnh báo dùng WAV cache riêng.
    """
    from google.genai import types
    prompt = ('Bạn là trợ lý dẫn đường cho người khiếm thị trong nhà. Mô tả TỐI ĐA 2 câu ngắn, '
              'đúng sự thật: vật gần nhất ở đâu (trái/phải/trước, gần/xa) và điều gì cần chú ý '
              'an toàn ngay lập tức. Không bịa chi tiết. Trả lời tiếng Việt bằng '
              'câu hoàn chỉnh và kết thúc bằng dấu câu. Không kết thúc câu bằng '
              'từ nối hoặc động từ phụ như "đang", "sẽ", "và". Yêu cầu: ' + question)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level='LOW'),
        response_mime_type='application/json',
        response_schema={
            'type': 'OBJECT',
            'required': ['answer'],
            'properties': {
                'answer': {
                    'type': 'STRING',
                    'description': 'Một hoặc hai câu tiếng Việt hoàn chỉnh.',
                },
            },
        },
    )
    with GEMINI_LOCK:
        response = get_gemini_client().models.generate_content(
            model=DESCRIBE_MODEL, contents=[prompt, image], config=config)
    text = extract_final_text(response)
    if description_is_incomplete(response, text):
        raise ValueError(f'incomplete response from Gemini: {text!r}')
    return text


def transcribe(speech, sample_rate: int):
    """STT + độ tin cậy (xác suất trung bình trên token sinh ra, thang 0..1)."""
    import torch
    p = MODEL_STATE['stt_processor']; m = MODEL_STATE['stt']
    x = p(speech, sampling_rate=sample_rate, return_tensors='pt').input_features.to('cuda:0', dtype=torch.float16)
    with STT_LOCK, torch.inference_mode():
        gen = m.generate(x, language='vi', task='transcribe', max_new_tokens=96,
                         do_sample=False, num_beams=1,
                         return_dict_in_generate=True, output_scores=True)
    text = p.batch_decode(gen.sequences, skip_special_tokens=True)[0].strip()
    n_prompt = gen.sequences.shape[1] - len(gen.scores)
    gen_ids = gen.sequences[:, n_prompt:].cpu()
    scores = torch.stack(gen.scores, dim=1).float().cpu()
    lp = torch.log_softmax(scores, dim=-1).gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)[0]
    real = gen_ids[0] != p.tokenizer.eos_token_id
    conf = float(lp[real].mean().exp()) if real.any() else 0.0
    return text, conf


def synthesize_wav(text: str) -> bytes:
    """Serialize CPU TTS calls without blocking the asyncio event loop."""
    with TTS_LOCK:
        audio = MODEL_STATE['tts'].infer(text, voice_id=cfg.TTS_VOICE)
        with tempfile.NamedTemporaryFile(suffix='.wav') as output:
            MODEL_STATE['tts'].save(audio, output.name)
            return Path(output.name).read_bytes()


def with_cached_reply_audio(route: dict) -> dict:
    """Attach a same-origin WAV URL only when that cache entry exists."""
    result = dict(route)
    code = result.get('reply_code')
    path = Path(cfg.ALERT_AUDIO_DIR) / f'{code}.wav' if code else None
    result['reply_audio_url'] = (
        f'/audio/{code}.wav' if path is not None and path.is_file() else None
    )
    return result


def app_factory():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from starlette.staticfiles import StaticFiles

    def require_ready():
        if not MODEL_STATE.get('ready'):
            raise HTTPException(503, MODEL_STATE.get('error') or 'models are still loading')


    @asynccontextmanager
    async def lifespan(_app):
        MODEL_STATE.setdefault('ready', False)
        threading.Thread(target=_load_models_worker, daemon=True, name='model-loader').start()
        yield

    app = FastAPI(title='Indoor AI Assistant API', version='0.2', lifespan=lifespan)

    @app.get('/health')
    async def health():
        return {
            'status': 'ok' if MODEL_STATE.get('ready') else ('error' if MODEL_STATE.get('error') else 'loading'),
            'error': MODEL_STATE.get('error'),
            'models': {'vlm': cfg.VLM_MODEL, 'stt': cfg.STT_MODEL, 'tts_voice': cfg.TTS_VOICE,
                       'depth': 'Depth-Anything-V2-Metric-Indoor-Large-hf', 'describe': DESCRIBE_MODEL},
            'stt_confidence_threshold': STT_CONFIDENCE_THRESHOLD,
        }

    @app.get('/v1/alerts')
    async def alerts_manifest():
        """Danh sách câu cảnh báo critical -> frontend preload WAV cache (§9)."""
        return {'phrases': cfg.ALERT_PHRASES,
                'audio_url_pattern': '/audio/{code}.wav',
                'audio_kind': 'emergency',
                'audio_priority': cfg.VOICE_AUDIO_PRIORITIES['emergency']}

    @app.get('/v1/bootstrap')
    async def bootstrap():
        """Cấu hình voice-first ổn định để frontend không hard-code hội thoại."""
        return {
            'default_mode': 'waiting',
            'welcome': {
                'reply_code': 'WELCOME',
                'reply_text': cfg.SYSTEM_PHRASES['WELCOME'],
                'audio_kind': cfg.SYSTEM_PHRASE_KINDS['WELCOME'],
                'audio_priority': cfg.VOICE_AUDIO_PRIORITIES['confirmation'],
            },
            'system_phrases': cfg.SYSTEM_PHRASES,
            'system_audio': {
                code: (
                    f'/audio/{code}.wav'
                    if (Path(cfg.ALERT_AUDIO_DIR) / f'{code}.wav').is_file()
                    else None
                )
                for code in cfg.SYSTEM_PHRASES
            },
            'audio_priorities': cfg.VOICE_AUDIO_PRIORITIES,
            'voice_capture': {
                'silence_timeout_ms': 800,
                'max_utterance_ms': 8000,
                'resume_after_playback_ms': 300,
            },
        }

    @app.post('/v1/intent')
    async def intent(payload: dict):
        """Route text đã nhận diện; hữu ích cho test và frontend typed client."""
        text = str(payload.get('text', '')).strip()
        if not text:
            raise HTTPException(400, 'text is required')
        return with_cached_reply_audio(
            cfg.route_voice_intent(
                text,
                str(payload.get('current_mode', 'waiting')),
                str(payload.get('resume_mode', 'guide')),
            )
        )

    app.mount('/audio', StaticFiles(directory=cfg.ALERT_AUDIO_DIR), name='audio')

    from fastapi.responses import FileResponse

    @app.get('/')
    async def index():
        page = ROOT / 'web' / 'dist' / 'index.html'
        return FileResponse(page if page.is_file() else ROOT / 'web' / 'index.html',
                            headers={'Cache-Control': 'no-store'})

    ui_assets = ROOT / 'web' / 'dist' / 'ui-assets'
    if ui_assets.is_dir():
        app.mount('/ui-assets', StaticFiles(directory=str(ui_assets)), name='ui-assets')

    @app.post('/v1/stt')
    async def stt(
        audio: UploadFile = File(...),
        current_mode: str = Form('waiting'),
        resume_mode: str = Form('guide'),
        turn_id: str = Form(''),
    ):
        data = await audio.read()
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, 'audio exceeds 10 MiB')
        require_ready()
        import librosa
        speech, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
        text, conf = await asyncio.to_thread(transcribe, speech, sr)
        if conf < STT_CONFIDENCE_THRESHOLD:
            wav = Path(cfg.ALERT_AUDIO_DIR) / 'REPEAT_PLEASE.wav'
            payload = {
                'turn_id': turn_id[:128] or None,
                'text': text,
                'confidence': round(conf, 4),
                'repeat': True,
                'intent': None,
                'mode': current_mode,
                'reply_code': 'REPEAT_PLEASE',
                'audio_kind': 'repeat',
                'audio_priority': cfg.VOICE_AUDIO_PRIORITIES['repeat'],
                'reply_audio_url': '/audio/REPEAT_PLEASE.wav',
            }
            if wav.exists():
                payload['audio_base64'] = base64.b64encode(wav.read_bytes()).decode()
            return JSONResponse(payload)
        return {
            'turn_id': turn_id[:128] or None,
            'text': text,
            'confidence': round(conf, 4),
            'repeat': False,
            **with_cached_reply_audio(
                cfg.route_voice_intent(text, current_mode, resume_mode)
            ),
        }

    @app.post('/v1/tts')
    async def tts(payload: dict):
        text = str(payload.get('text', '')).strip()
        if not text or len(text) > 500:
            raise HTTPException(400, 'text must be 1-500 characters')
        require_ready()
        code = next((k for k, v in {**cfg.ALERT_PHRASES, **cfg.SYSTEM_PHRASES}.items() if v == text), None)
        if code:
            path = Path(cfg.ALERT_AUDIO_DIR) / (code + '.wav')
            if path.exists():
                return JSONResponse({'message_code': code, 'cached': True, 'audio_base64': base64.b64encode(path.read_bytes()).decode()})
        wav = await asyncio.to_thread(synthesize_wav, text)
        return {
            'message_code': None,
            'cached': False,
            'audio_base64': base64.b64encode(wav).decode(),
        }

    @app.post('/v1/describe')
    async def describe(payload: dict):
        raw = payload.get('image_base64'); question = str(payload.get('question', 'Mô tả vật dụng trong ảnh.')).strip()
        request_id = str(payload.get('request_id', '')).strip()[:128] or None
        want_audio = bool(payload.get('tts', True))
        if not raw:
            raise HTTPException(400, 'image_base64 is required')
        if not os.getenv('GEMINI_API_KEY'):
            raise HTTPException(503, 'GEMINI_API_KEY is not configured')
        if want_audio:
            require_ready()
        try:
            image = prepare_image(base64.b64decode(raw)); text = await asyncio.to_thread(describe_with_api, image, question)
            result: dict = {
                'request_id': request_id,
                'model': DESCRIBE_MODEL,
                'provider': 'gemini',
                'text': text,
                'audio_kind': 'answer',
                'audio_priority': cfg.VOICE_AUDIO_PRIORITIES['answer'],
                'interruptible': True,
            }
            if want_audio:
                wav = await asyncio.to_thread(synthesize_wav, text)
                result['audio_base64'] = base64.b64encode(wav).decode()
            return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f'description provider failed: {exc}') from exc

    @app.websocket('/v1/vision')
    async def vision(ws: WebSocket):
        await ws.accept()
        session = fus.FusionSession()
        latest = asyncio.Queue(maxsize=1)
        dropped_frames = 0

        def process_frame(message, received_at, dropped):
            queue_wait_ms = round((time.perf_counter() - received_at) * 1000)
            raw = message.get('image_base64')
            if not raw:
                raise ValueError('image_base64 is required')
            result = infer_vision(prepare_image(base64.b64decode(raw)))
            depth_map = result.pop('_depth_map')
            ev_vlm = fus.vlm_evidence(result['vlm'])
            ev_geo = fus.depth_evidence(depth_map)
            decision = fus.decide_frame(ev_vlm, ev_geo, session.config)
            event = fus.update_session(
                session, decision, ev_vlm,
                new_semantics=result.get('vlm_fresh', True),
            )
            return {
                'ok': True, **result,
                'frame_id': message.get('frame_id'),
                'dropped_frames': dropped,
                'queue_wait_ms': queue_wait_ms,
                'geometry': {k: v for k, v in ev_geo.items() if k != 'per_col_rank'},
                'decision': event,
            }

        async def receive_latest():
            nonlocal dropped_frames
            try:
                while True:
                    item = (await ws.receive_json(), time.perf_counter())
                    if latest.full():
                        latest.get_nowait()
                        dropped_frames += 1
                    latest.put_nowait((*item, dropped_frames))
            except WebSocketDisconnect:
                return

        async def infer_latest():
            while True:
                message, received_at, dropped = await latest.get()
                try:
                    payload = await asyncio.to_thread(
                        process_frame, message, received_at, dropped
                    )
                except Exception as exc:
                    payload = {'ok': False, 'error': str(exc),
                               'frame_id': message.get('frame_id'),
                               'dropped_frames': dropped}
                payload['server_total_ms'] = round(
                    (time.perf_counter() - received_at) * 1000
                )
                await ws.send_json(payload)
        try:
            receiver = asyncio.create_task(receive_latest())
            worker = asyncio.create_task(infer_latest())
            done, pending = await asyncio.wait(
                {receiver, worker}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
    return app


app = app_factory()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8000')))
