#!/usr/bin/env python3
"""Indoor MVP backend API."""
import base64, io, json, os, sys, tempfile, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

ROOT = Path('/home/ubuntu/ai-assistant')
sys.path.insert(0, str(ROOT / 'scripts'))

from dotenv import load_dotenv

load_dotenv(ROOT / '.env')

import mvp_config as cfg
import fusion as fus

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
STT_CONFIDENCE_THRESHOLD = 0.60   # §5: dưới ngưỡng thì phát REPEAT_PLEASE, không đoán
MODEL_STATE: dict[str, Any] = {}
DESCRIBE_MODEL = os.getenv('DESCRIBE_MODEL', 'gemini-3.6-flash')


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


def describe_with_api(image, question):
    """Kênh thuyết minh (non-critical): Gemini thinking-LOW + giới hạn 2 câu.

    Đo được (logs/latency_e2e.json): prompt dài mặc định -> 19 s/end-to-end.
    Thinking LOW + câu ngắn -> ~3 s Gemini + ~0.7 s TTS. Kênh cảnh báo KHÔNG
    đi qua đây (đã có WAV cache 4 ms).
    """
    from google.genai import types
    prompt = ('Bạn là trợ lý dẫn đường cho người khiếm thị trong nhà. Mô tả TỐI ĐA 2 câu ngắn, '
              'đúng sự thật: vật gần nhất ở đâu (trái/phải/trước, gần/xa) và điều gì cần chú ý '
              'an toàn ngay lập tức. Không bịa chi tiết. Trả lời tiếng Việt. Yêu cầu: ' + question)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level='LOW'))
    response = get_gemini_client().models.generate_content(
        model=DESCRIBE_MODEL, contents=[prompt, image], config=config)
    text = (response.text or '').strip()
    if not text:
        raise ValueError('empty response from Gemini')
    return text


def transcribe(speech, sample_rate: int):
    """STT + độ tin cậy (xác suất trung bình trên token sinh ra, thang 0..1)."""
    import torch
    p = MODEL_STATE['stt_processor']; m = MODEL_STATE['stt']
    x = p(speech, sampling_rate=sample_rate, return_tensors='pt').input_features.to('cuda:0', dtype=torch.float16)
    with torch.inference_mode():
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


def app_factory():
    from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
                'audio_url_pattern': '/audio/{code}.wav'}

    app.mount('/audio', StaticFiles(directory=cfg.ALERT_AUDIO_DIR), name='audio')

    from fastapi.responses import FileResponse

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'web' / 'index.html')

    @app.post('/v1/stt')
    async def stt(audio: UploadFile = File(...)):
        data = await audio.read()
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, 'audio exceeds 10 MiB')
        require_ready()
        import librosa
        speech, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
        text, conf = transcribe(speech, sr)
        if conf < STT_CONFIDENCE_THRESHOLD:
            wav = Path(cfg.ALERT_AUDIO_DIR) / 'REPEAT_PLEASE.wav'
            payload = {'text': text, 'confidence': round(conf, 4), 'repeat': True}
            if wav.exists():
                payload['audio_base64'] = base64.b64encode(wav.read_bytes()).decode()
            return JSONResponse(payload)
        return {'text': text, 'confidence': round(conf, 4), 'repeat': False}

    @app.post('/v1/tts')
    async def tts(payload: dict):
        text = str(payload.get('text', '')).strip()
        if not text or len(text) > 500:
            raise HTTPException(400, 'text must be 1-500 characters')
        require_ready()
        code = next((k for k, v in cfg.ALERT_PHRASES.items() if v == text), None)
        if code:
            path = Path(cfg.ALERT_AUDIO_DIR) / (code + '.wav')
            if path.exists():
                return JSONResponse({'message_code': code, 'cached': True, 'audio_base64': base64.b64encode(path.read_bytes()).decode()})
        audio = MODEL_STATE['tts'].infer(text, voice_id=cfg.TTS_VOICE)
        with tempfile.NamedTemporaryFile(suffix='.wav') as output:
            MODEL_STATE['tts'].save(audio, output.name)
            return {'message_code': None, 'cached': False, 'audio_base64': base64.b64encode(Path(output.name).read_bytes()).decode()}

    @app.post('/v1/describe')
    async def describe(payload: dict):
        raw = payload.get('image_base64'); question = str(payload.get('question', 'Mô tả vật dụng trong ảnh.')).strip()
        want_audio = bool(payload.get('tts', True))
        if not raw:
            raise HTTPException(400, 'image_base64 is required')
        if not os.getenv('GEMINI_API_KEY'):
            raise HTTPException(503, 'GEMINI_API_KEY is not configured')
        if want_audio:
            require_ready()
        try:
            image = prepare_image(base64.b64decode(raw)); text = describe_with_api(image, question)
            result: dict = {'model': DESCRIBE_MODEL, 'text': text}
            if want_audio:
                audio = MODEL_STATE['tts'].infer(text, voice_id=cfg.TTS_VOICE)
                with tempfile.NamedTemporaryFile(suffix='.wav') as output:
                    MODEL_STATE['tts'].save(audio, output.name)
                    result['audio_base64'] = base64.b64encode(Path(output.name).read_bytes()).decode()
            return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f'description provider failed: {exc}') from exc

    @app.websocket('/v1/vision')
    async def vision(ws: WebSocket):
        await ws.accept()
        session = fus.FusionSession()
        try:
            while True:
                message = await ws.receive_json(); raw = message.get('image_base64')
                if not raw:
                    await ws.send_json({'error': 'image_base64 is required'}); continue
                try:
                    result = infer_vision(prepare_image(base64.b64decode(raw)))
                    depth_map = result.pop('_depth_map')
                    ev_vlm = fus.vlm_evidence(result['vlm'])
                    ev_geo = fus.depth_evidence(depth_map)
                    decision = fus.decide_frame(ev_vlm, ev_geo, session.config)
                    event = fus.update_session(session, decision, ev_vlm)
                    await ws.send_json({'ok': True, **result,
                                        'geometry': {k: v for k, v in ev_geo.items() if k != 'per_col_rank'},
                                        'decision': event})
                except Exception as exc:
                    await ws.send_json({'ok': False, 'error': str(exc)})
        except WebSocketDisconnect:
            pass
    return app


app = app_factory()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.getenv('HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8000')))
