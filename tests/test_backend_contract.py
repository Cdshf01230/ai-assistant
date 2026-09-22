"""Exercise both API entrypoints without GPU/model loading."""
import base64
import importlib.util
import json
import os
import re
import secrets
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mvp_config as cfg
import fusion as fus

class BackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg.ALERT_AUDIO_DIR = str(ROOT / "assets" / "audio")
        spec = importlib.util.spec_from_file_location("review_api", ROOT / "api_main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.MODEL_STATE["ready"] = True
        module.prepare_image = lambda data: data
        module.describe_with_api = lambda image, question: "Có ghế phía trước."
        module.synthesize_wav = lambda text: b"test-audio"
        module.transcribe = lambda *args: ("dẫn đường", .95)
        cls.module = module
        notebook = json.loads((ROOT / "kaggle_ai_assistant.ipynb").read_text(encoding="utf-8"))
        cell = next("".join(c["source"]) for c in notebook["cells"]
                    if "# 5. FastAPI" in "".join(c.get("source", [])))
        cls.notebook_api_cell = cell
        cls.ns = {
            "secrets": secrets, "cfg": cfg, "fus": fus,
            "WORKDIR": ROOT, "audio_dir": ROOT / "assets" / "audio",
            "MODEL_STATE": {"ready": True}, "DESCRIBE_MODEL": "test",
            "MAX_AUDIO_BYTES": 10*1024*1024, "MAX_IMAGE_BYTES": 5*1024*1024,
            "STT_CONFIDENCE_THRESHOLD": .60, "base64": base64, "io": __import__("io"),
            "time": time,
            "os": os, "prepare_image": lambda data: data,
            "decode_base64": lambda value, limit: base64.b64decode(value),
            "new_vision_state": lambda: {},
            "transcribe": lambda *args: ("dẫn đường", .95),
            "synthesize_wav": lambda text: b"test-audio",
            "describe_with_gemini": lambda image, question: "Có ghế phía trước.",
        }
        exec(compile(cell, "notebook-api-cell", "exec"), cls.ns)
        cls.clients = [TestClient(module.app, base_url="https://testserver"),
                       TestClient(cls.ns["app"], base_url="https://testserver")]
        cls.clients[1].get("/", params={"token": cls.ns["ACCESS_TOKEN"]})

    @classmethod
    def tearDownClass(cls):
        for client in cls.clients:
            client.close()

    def test_static_build_and_bootstrap(self):
        self.assertEqual(self.module.DESCRIBE_MODEL, "gemini-3.5-flash-lite")
        for client in self.clients:
            with self.subTest(client=client):
                page = client.get("/")
                self.assertEqual(page.status_code, 200)
                self.assertIn('lang="vi"', page.text)
                files = re.findall(r'(?:src|href)="(/ui-assets/[^"]+)"', page.text)
                self.assertGreaterEqual(len(files), 2)
                for path in files:
                    self.assertEqual(client.get(path).status_code, 200)
                data = client.get("/v1/bootstrap").json()
                self.assertIn("WELCOME", data["system_phrases"])
                for url in data["system_audio"].values():
                    if url:
                        self.assertEqual(client.get(url).status_code, 200)

    def test_intent_and_pause_resume(self):
        for client in self.clients:
            route = client.post("/v1/intent", json={"text":"tạm dừng","current_mode":"narration"}).json()
            self.assertEqual(route["mode"], "paused")
            resumed = client.post("/v1/intent", json={
                "text":"tiếp tục","current_mode":"paused","resume_mode":route["resume_mode"]}).json()
            self.assertEqual(resumed["mode"], "narration")

    def test_stt_legacy_and_new_fields_low_confidence(self):
        fake_audio = types.SimpleNamespace(load=lambda *args, **kwargs: ([0]*16000,16000))
        with patch.dict(sys.modules, {"librosa":fake_audio}):
            for client in self.clients:
                result = client.post("/v1/stt", files={"audio":("q.wav",b"audio","audio/wav")},
                                     data={"turn_id":"turn-7","current_mode":"waiting"}).json()
                self.assertEqual(result["turn_id"], "turn-7")
                self.assertEqual(result["mode"], "guide")
                legacy = client.post("/v1/stt",files={"audio":("q.wav",b"audio","audio/wav")})
                self.assertEqual(legacy.status_code,200)
            self.module.transcribe = self.ns["transcribe"] = lambda *args: ("dẫn đường", .1)
            try:
                for client in self.clients:
                    result = client.post("/v1/stt",files={"audio":("q.wav",b"audio","audio/wav")},
                                         data={"current_mode":"narration"}).json()
                    self.assertTrue(result["repeat"])
                    self.assertIsNone(result["intent"])
                    self.assertEqual(result["mode"], "narration")
            finally:
                self.module.transcribe = self.ns["transcribe"] = lambda *args: ("dẫn đường", .95)

    def test_describe_echoes_id(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-placeholder"}):
            for client in self.clients:
                result=client.post("/v1/describe",json={
                    "image_base64":"YWJj","question":"có gì","request_id":"query-9","tts":False})
                self.assertEqual(result.status_code,200)
                self.assertEqual(result.json()["request_id"],"query-9")
                self.assertEqual(result.json()["audio_priority"],40)

    def test_rejects_truncated_gemini_description(self):
        finish = types.SimpleNamespace(name="MAX_TOKENS")
        response = types.SimpleNamespace(
            candidates=[types.SimpleNamespace(finish_reason=finish)]
        )
        self.assertTrue(self.module.description_is_incomplete(
            response, "Phía trước bạn"
        ))
        complete = types.SimpleNamespace(
            candidates=[types.SimpleNamespace(
                finish_reason=types.SimpleNamespace(name="STOP")
            )]
        )
        self.assertFalse(self.module.description_is_incomplete(
            complete, "Phía trước bạn là một bàn phím."
        ))
        self.assertTrue(self.module.description_is_incomplete(
            complete, "Một người ở phía bên phải đang."
        ))
        response_with_thought = types.SimpleNamespace(candidates=[
            types.SimpleNamespace(content=types.SimpleNamespace(parts=[
                types.SimpleNamespace(text="đang suy nghĩ", thought=True),
                types.SimpleNamespace(
                    text="Đây là một bàn phím.", thought=False
                ),
            ]))
        ])
        self.assertEqual(
            self.module.extract_final_text(response_with_thought),
            "Đây là một bàn phím.",
        )
        notebook = json.loads(
            (ROOT / "kaggle_ai_assistant.ipynb").read_text(encoding="utf-8")
        )
        inference = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if "# 4. Inference" in "".join(cell.get("source", []))
        )
        self.assertIn("extract_final_text", inference)
        self.assertIn("description_is_incomplete", inference)
        self.assertEqual(
            inference.count("_GEMINI_CLIENT.models.generate_content("), 1
        )

    def test_notebook_assets_and_api_require_auth(self):
        with TestClient(self.ns["app"],base_url="https://testserver") as anonymous:
            for path in ("/", "/v1/bootstrap"):
                self.assertEqual(anonymous.get(path).status_code,401)
            page=self.clients[1].get("/")
            asset=re.search(r'src="(/ui-assets/[^"]+)"',page.text).group(1)
            self.assertEqual(anonymous.get(asset).status_code,401)

    def test_vision_uses_latest_frame_queue(self):
        self.assertIn("asyncio.Queue(maxsize=1)", self.notebook_api_cell)
        self.assertIn("receive_latest", self.notebook_api_cell)
        self.assertIn("infer_latest", self.notebook_api_cell)

        original_api = self.module.infer_vision
        original_notebook = self.ns.get("infer_vision_stream")
        try:
            for index, client in enumerate(self.clients):
                with self.subTest(entrypoint=index):
                    self._assert_latest_frame_queue(index, client)
        finally:
            self.module.infer_vision = original_api
            if original_notebook is None:
                self.ns.pop("infer_vision_stream", None)
            else:
                self.ns["infer_vision_stream"] = original_notebook

    def _assert_latest_frame_queue(self, index, client):
                started, release = threading.Event(), threading.Event()
                calls = 0

                def fake_infer(*_args):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        started.set()
                        release.wait(timeout=2)
                    return {"latency_ms": 1, "vlm": None, "_depth_map": object()}

                if index == 0:
                    self.module.infer_vision = fake_infer
                else:
                    self.ns["infer_vision_stream"] = fake_infer
                with patch.object(fus, "vlm_evidence", return_value={}), \
                     patch.object(fus, "depth_evidence", return_value={"free_min": 1}), \
                     patch.object(fus, "decide_frame", return_value={}), \
                     patch.object(fus, "update_session", return_value={"alert": False}):
                    path = "/v1/vision"
                    if index == 1:
                        path += "?token=" + self.ns["ACCESS_TOKEN"]
                    with client.websocket_connect(path) as ws:
                        ws.send_json({"frame_id": 1, "image_base64": "YQ=="})
                        self.assertTrue(started.wait(timeout=1))
                        ws.send_json({"frame_id": 2, "image_base64": "Yg=="})
                        ws.send_json({"frame_id": 3, "image_base64": "Yw=="})
                        time.sleep(.05)
                        release.set()
                        first = ws.receive_json()
                        latest = ws.receive_json()
                        self.assertEqual(first["frame_id"], 1)
                        self.assertEqual(latest["frame_id"], 3)
                        self.assertGreaterEqual(latest["dropped_frames"], 1)
                        self.assertIn("server_total_ms", latest)
                        ws.close()

if __name__ == "__main__":
    unittest.main()
