#!/usr/bin/env python3
"""Đo latency end-to-end qua đúng đường đi của demo (TLS/WSS trên server thật).

Ngân sách §12: <300 ms rất tốt | 300-600 tốt | 600-1000 demo được | >1500 không
dùng được cho cảnh báo vật cản gần.

Chuỗi đo: encode JPEG (như browser) -> WSS vision (RTT + phần server tự báo)
-> STT round-trip -> TTS cached vs động -> describe. Không có điện thoại thật
nên T_camera và thời gian phát audio phía client không đo được — ghi rõ.
"""
from __future__ import annotations

import base64, io, json, ssl, statistics, sys, time
from pathlib import Path

import urllib.request

ROOT = Path('/home/ubuntu/ai-assistant')
BASE = 'https://127.0.0.1:8443'
WSS = 'wss://127.0.0.1:8443/v1/vision'
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

LONG_EDGE, JPEG_Q = 640, 80   # PIL dùng thang 1-95; browser canvas là 0..1


def pct(xs, p):
    xs = sorted(xs); return xs[min(int(len(xs) * p), len(xs) - 1)]


def line(name, xs):
    print(f"  {name:<34} median {statistics.median(xs):7.0f}  p95 {pct(xs, .95):7.0f}  min {min(xs):6.0f}  max {max(xs):6.0f}  n={len(xs)}")


def main() -> int:
    from PIL import Image
    import websockets

    frames = ['pole_00', 'synth_person_front', 'clearpath_00', 'crowd_00']
    jpegs = []
    t_enc = []
    for f in frames:
        img = Image.open(ROOT / f'assets/frames/{f}.jpg').convert('RGB')
        w, h = img.size
        s = LONG_EDGE / max(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=JPEG_Q)
        raw = buf.getvalue()
        b64 = base64.b64encode(raw).decode()
        jpegs.append(b64)

    # --- chờ model sẵn sàng ---
    for _ in range(80):
        st = json.load(urllib.request.urlopen(BASE + '/health', context=CTX, timeout=5))
        if st['status'] == 'ok':
            break
        time.sleep(4)

    async def bench():
        lat_rtt, lat_server, first = [], [], None
        async with websockets.connect(WSS, ssl=CTX, max_size=20 << 20) as ws:
            # frame đầu là warm-up (CUDA kernel, cache KV) — bỏ khỏi thống kê
            for i in range(24):
                b64 = jpegs[i % len(jpegs)]
                t0 = time.perf_counter()
                await ws.send(json.dumps({'image_base64': b64}))
                r = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                rtt = (time.perf_counter() - t0) * 1000
                if i == 0:
                    first = rtt
                    continue
                lat_rtt.append(rtt)
                lat_server.append(r.get('latency_ms', 0))
        return lat_rtt, lat_server, first

    import asyncio
    print('== Vision WSS (frame 640/q80, bỏ warm-up) ==')
    lat_rtt, lat_server, first = asyncio.run(bench())
    print(f'  {"warm-up (frame đầu)":<34} {first:.0f} ms — không tính')
    line('RTT send->decision', lat_rtt)
    line('trong đó server (VLM+depth)', lat_server)
    overhead = [a - b for a, b in zip(lat_rtt, lat_server)]
    line('upload+TLS+JSON+queue', overhead)

    # --- STT ---
    print('== STT /v1/stt (wav ~2-3 s, push-to-talk) ==')
    lat_stt = []
    for _ in range(6):
        data = (ROOT / 'assets/tts_test/command_00.wav').read_bytes()
        boundary = '----x'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; '
                f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
        req = urllib.request.Request(BASE + '/v1/stt', data=body, method='POST',
                                     headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        t0 = time.perf_counter()
        urllib.request.urlopen(req, context=CTX, timeout=120).read()
        lat_stt.append((time.perf_counter() - t0) * 1000)
    line('STT round-trip', lat_stt[1:])   # bỏ warm-up

    # --- TTS ---
    print('== TTS /v1/tts ==')
    lat_tts_c, lat_tts_d = [], []
    for _ in range(5):
        req = urllib.request.Request(BASE + '/v1/tts',
                                     data=json.dumps({'text': 'Dừng lại.'}).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.perf_counter()
        urllib.request.urlopen(req, context=CTX, timeout=60).read()
        lat_tts_c.append((time.perf_counter() - t0) * 1000)
    for _ in range(3):
        req = urllib.request.Request(BASE + '/v1/tts',
                                     data=json.dumps({'text': 'Cột điện cách bạn khoảng ba bước về phía trái.'}).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.perf_counter()
        urllib.request.urlopen(req, context=CTX, timeout=120).read()
        lat_tts_d.append((time.perf_counter() - t0) * 1000)
    line('cached WAV (critical path)', lat_tts_c[1:])
    line('dynamic CPU/ONNX (câu dài)', lat_tts_d[1:])

    # --- describe ---
    print('== Describe /v1/describe (Gemini + TTS động) ==')
    img_b64 = jpegs[0]
    lat_desc = []
    for q in ['Phía trước tôi có gì?', 'Mô tả xung quanh.', 'Có cửa nào gần không?']:
        req = urllib.request.Request(BASE + '/v1/describe',
                                     data=json.dumps({'image_base64': img_b64, 'question': q}).encode(),
                                     headers={'Content-Type': 'application/json'})
        t0 = time.perf_counter()
        urllib.request.urlopen(req, context=CTX, timeout=180).read()
        lat_desc.append((time.perf_counter() - t0) * 1000)
    line('describe (Gemini+TTS, n=3)', lat_desc)

    # --- tổng hợp theo ngân sách §12 ---
    med = statistics.median(lat_rtt)
    band = ('rất tốt' if med < 300 else 'tốt' if med < 600 else
            'demo được' if med <= 1000 else 'KHÔNG dùng được cho cảnh báo gần')
    warn_chain = med + 100   # + phát WAV phía client ~100ms (ước lượng, chưa đo trên máy thật)
    result = {
        'vision_rtt_median_ms': round(med), 'vision_rtt_p95_ms': round(pct(lat_rtt, .95)),
        'vision_server_median_ms': round(statistics.median(lat_server)),
        'overhead_median_ms': round(statistics.median(overhead)),
        'stt_median_ms': round(statistics.median(lat_stt[1:])),
        'tts_cached_median_ms': round(statistics.median(lat_tts_c[1:])),
        'tts_dynamic_median_ms': round(statistics.median(lat_tts_d[1:])),
        'describe_median_ms': round(statistics.median(lat_desc)),
        'budget_band_vision_only': band,
        'warn_chain_estimate_client_play_ms': round(warn_chain),
        'note': 'Chưa gồm T_camera và latency phát audio trên thiết bị thật.',
    }
    print('\n== Tổng hợp theo ngân sách §12 ==')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out = ROOT / 'logs/latency_e2e.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print('->', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
