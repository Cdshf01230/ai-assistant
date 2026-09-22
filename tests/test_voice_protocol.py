#!/usr/bin/env python3
"""Unit test cho voice intent protocol, không cần GPU hay model."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mvp_config as cfg  # noqa: E402
import fusion  # noqa: E402


class VoiceProtocolTests(unittest.TestCase):
    def route(self, text, mode="waiting", resume="guide"):
        return cfg.route_voice_intent(text, mode, resume)

    def test_normalizes_vietnamese_accents(self):
        self.assertEqual(cfg.normalize_voice_text("  DẪN ĐƯỜNG! "), "dan duong")

    def test_switches_to_guide(self):
        result = self.route("bật chế độ dẫn đường")
        self.assertEqual(result["intent"], "switch_mode")
        self.assertEqual(result["mode"], "guide")
        self.assertFalse(result["should_describe"])

    def test_plain_narration_command_does_not_call_describe(self):
        for text in ("thuyết minh", "bật chế độ thuyết minh"):
            with self.subTest(text=text):
                result = self.route(text)
                self.assertEqual(result["intent"], "switch_mode")
                self.assertEqual(result["mode"], "narration")
                self.assertFalse(result["should_describe"])

    def test_question_in_narration_calls_describe(self):
        result = self.route("phía trước có gì", "narration")
        self.assertEqual(result["intent"], "ask_description")
        self.assertTrue(result["should_describe"])
        self.assertEqual(result["question"], "phía trước có gì")

    def test_direct_description_command_calls_describe(self):
        result = self.route("mô tả người phía trước cho tôi")
        self.assertEqual(result["mode"], "narration")
        self.assertTrue(result["should_describe"])

    def test_pause_and_resume_restore_previous_mode(self):
        paused = self.route("tạm dừng", "narration")
        self.assertEqual(paused["mode"], "paused")
        self.assertEqual(paused["resume_mode"], "narration")
        resumed = self.route("tiếp tục", "paused", paused["resume_mode"])
        self.assertEqual(resumed["mode"], "narration")

    def test_repeat_has_no_new_spoken_phrase(self):
        result = self.route("lặp lại", "guide")
        self.assertEqual(result["intent"], "repeat_last")
        self.assertIsNone(result["reply_code"])
        self.assertIsNone(result["reply_text"])

    def test_command_words_inside_question_do_not_switch_mode(self):
        for text in ("mô tả biển dừng lại phía trước", "biển chỉ đường viết gì",
                     "không tạm dừng"):
            self.assertEqual(self.route(text, "narration")["intent"], "ask_description")

    def test_polite_mode_switch_is_not_a_question(self):
        self.assertEqual(self.route("chuyển sang chế độ thuyết minh")["intent"], "switch_mode")

    def test_resume_while_active_preserves_mode(self):
        self.assertEqual(self.route("tiếp tục", "narration", "guide")["mode"], "narration")


class FusionSafetyTests(unittest.TestCase):
    @staticmethod
    def evidence(obj="person", position="front", action="slow"):
        return fusion.vlm_evidence({
            "hazard": True, "type": obj, "position": position,
            "distance": "near", "action": action,
        })

    def test_cached_qwen_result_does_not_confirm_person(self):
        session = fusion.FusionSession()
        decision = {"frame_positive": True, "risk": .8, "direction_ok": False}
        person = self.evidence()
        results = [
            fusion.update_session(session, decision, person, new_semantics=True),
            fusion.update_session(session, decision, person, new_semantics=False),
            fusion.update_session(session, decision, person, new_semantics=False),
            fusion.update_session(session, decision, person, new_semantics=True),
        ]
        self.assertFalse(any(item["alert"] for item in results))
        confirmed = fusion.update_session(
            session, decision, person, new_semantics=True
        )
        self.assertTrue(confirmed["alert"])
        self.assertEqual(confirmed["message_code"], "PERSON_AHEAD")

    def test_fresh_clear_breaks_person_confirmation_streak(self):
        session = fusion.FusionSession()
        positive = {"frame_positive": True, "risk": .8, "direction_ok": False}
        clear = {"frame_positive": False, "reason": "vlm_clear"}
        person = self.evidence()
        fusion.update_session(session, positive, person, new_semantics=True)
        fusion.update_session(session, clear, {"valid": True, "hazard": False},
                              new_semantics=True)
        after_clear = [
            fusion.update_session(session, positive, person, new_semantics=True),
            fusion.update_session(session, positive, person, new_semantics=True),
        ]
        self.assertFalse(any(item["alert"] for item in after_clear))
        confirmed = fusion.update_session(
            session, positive, person, new_semantics=True
        )
        self.assertTrue(confirmed["alert"])

    def test_direction_requires_vlm_and_depth_agreement(self):
        config = fusion.FusionConfig()
        geo_left = {"valid": True, "free_min": 3.0, "near_rank": .9,
                    "coverage": 1.0, "col": 0}
        wrong = self.evidence("pole", "front_left", "move_left")
        decision = fusion.decide_frame(wrong, geo_left, config)
        session = fusion.FusionSession(config=config)
        fusion.update_session(session, decision, wrong, new_semantics=True)
        result = fusion.update_session(session, decision, wrong, new_semantics=True)
        self.assertEqual(result["message_code"], "SLOW_DOWN")
        self.assertEqual(result["action"], "slow")


if __name__ == "__main__":
    unittest.main()
