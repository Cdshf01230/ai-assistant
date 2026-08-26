#!/usr/bin/env python3
"""Evidence fusion tổng quát: VLM (ngữ nghĩa) + depth (hình học) + trạng thái thời gian.

Phân công theo đúng chỗ mỗi nguồn mạnh — tất cả đều là SỐ ĐO, không phải phán đoán:

  VLM   quyết CÓ VẬT GÌ / Ở ĐÂU / NÊN LÀM GÌ
        (đúng tên vật 85%, đúng hướng 85-92% — logs/eval_vlm_qwen3vl4b.json)
        và MÙ HOÀN TOÀN khoảng cách: cùng một cây cột ở chân trời và ngay trước
        mặt trả y một đáp án, nhãn FAR không bao giờ được dùng trên 27 frame.
        => Không bao giờ để nhãn NEAR/MID/FAR của VLM quyết định cảnh báo.

  Depth quyết LỐI ĐI CÒN LÙI RA XA ĐƯỢC KHÔNG (tín hiệu hình học VLM không có).
        Thang mét của checkpoint metric sai 4-5 lần trên camera cầm tay (train
        VKITTI/Hypersim — logs/depth_round2.md) => CHỈ dùng tín hiệu THỨ TỰ
        trong khung hình (rank), cấm ngưỡng theo mét tuyệt đối.

  Chiến lược đã đo (12_fuse_vlm_depth.py): AND-thuần + cửa thoát cho các lớp
  depth mù cấu trúc (step/hole/door) giữ recall 100%, kéo specificity từ 58%
  lên 75-92% tuỳ checkpoint, âm khó 0% -> 25-75%.

Nguyên tắc tổng quát hoá:
  1. Mỗi nguồn trả Evidence {valid, score, ...}. Nguồn hỏng tự rút khỏi quyết
     định, không được tính thành "không có vật" (thà im hơn đoán — §16).
  2. Không hằng số nào gắn với lớp vật, trừ bảng BLIND_TYPES khai báo ở đây —
     đó là lỗ CẤU TRÚC của free-space scan, không phải tham số chỉnh nắn.
  3. Mọi ngưỡng nằm trong FusionConfig, kèm nơi đã đo. Đổi camera/độ cao/góc
     chúc là phải hiệu chuẩn lại (chạy lại benchmark_geometry_public.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_path as dpath

# --- Ngưỡng hành động -------------------------------------------------------
ACTION_SEVERITY = {"none": 0, "slow": 1, "move_left": 1, "move_right": 1, "stop": 2}

# Bảng điểm mù của free-space scan (xem depth_path.DEPTH_BLIND_TYPES):
# step/hole không chắn tia nhìn, door kính cho tia xuyên qua. Depth đọc thấy
# "trống phía sau" là SAI đối với ba lớp này => không được phủ quyết VLM.
BLIND_TYPES = frozenset(dpath.DEPTH_BLIND_TYPES)


@dataclass(frozen=True)
class FusionConfig:
    """Mọi ngưỡng của fusion. Mỗi hằng số ghi rõ nguồn số đo."""

    # Ngưỡng tuyệt đối ĐÃ HIỆU CHUẨN cho cặp (checkpoint depth, camera) đang dùng.
    # Hiệu chuẩn bằng benchmarks/calibrate_depth_threshold.py trên dump
    # logs/depth_indoor_large.json (27 frame có nhãn): quét ngưỡng giữ recall 100%
    # rồi lấy specificity cao nhất -> 7.0 (spec 75%, âm khó 25%).
    # ⚠️ ĐỔI CHECKPOINT / CAMERA / ĐỘ CAO LÀ PHẢI HIỆU CHUẨN LẠI — thang mét của
    # checkpoint sai 4-5 lần so với thật (logs/depth_round2.md).
    free_abs_threshold: float = 7.0

    # Tín hiệu SCALE-FREE bổ sung: rank "bị chắn sớm" trong khung hình
    # near_rank = 1 - free_min/free_extent. Bắt được vật chắn mềm kiểu người
    # đứng giữa lối đi mà ngưỡng tuyệt đối bỏ sót khi phòng nhỏ
    # (synth_person_front: rank 0.68). indoor_demo_pipeline.py đã dùng 0.5.
    near_rank_threshold: float = 0.5

    # DEPTH EMERGENCY OVERRIDE — cảnh báo STOP kể cả khi VLM nói CLEAR.
    # Nguồn số đo: thẩm định tay 45 frame lỗi COCO (logs/coco_adjudication.md)
    # tìm ra lớp lỗi duy nhất còn lại: VLM trả CLEAR khi vật chắn <3 m trong nhà
    # (8 case, 6 case có free 0.79-1.67). Khoảng hở với frame trống thưa nhất
    # của bộ 27 frame là 2.39 (synthclear_indoor) -> chọn 2.0 làm điểm giữa.
    # 11/11 frame COCO âm có free<2.0 đều là cảnh báo ĐÚNG theo chuẩn POV
    # (bề mặt 0.3-1.7 m ngay trước mặt). ⚠️ Gắn với checkpoint+640px hiện tại;
    # đổi camera/checkpoint là phải hiệu chuẩn lại cùng quy trình.
    depth_emergency_threshold: float = 2.0

    # Trễ thời gian — indoor_demo_pipeline_full.json chạy confirm=2/clear=3.
    confirm_frames: int = 2
    clear_frames: int = 3

    # Khi depth hỏng hoàn toàn (NaN/không đủ pixel): 'trust_vlm_near' = chỉ tin
    # nhãn NEAR của VLM (recall-first, depth_probe_run.md: "coi depth=nan là gần");
    # 'silent' = bỏ frame (depth_path.should_warn_fused). An toàn > im lặng.
    depth_fail_policy: str = "trust_vlm_near"

    # Cross-check hướng: cột hình học (trái/giữa/phải) trái với position của VLM
    # thì hạ risk 20% và chặn escalation STOP — mâu thuẫn là thông tin.
    mismatch_risk_penalty: float = 0.8

    # Thuyết minh môi trường: tối thiểu cách nhau bao lâu, và chỉ khi đổi cảnh.
    narrate_min_interval_s: float = 10.0


@dataclass
class FusionSession:
    """Trạng thái theo phiên kết nối (một người dùng / một WebSocket)."""

    config: FusionConfig = field(default_factory=FusionConfig)
    active: bool = False
    positive_streak: int = 0
    clear_streak: int = 0
    last_key: tuple[str, str] | None = None
    last_severity: int = 0
    _last_narrate_t: float = float("-inf")
    _last_narrate_key: tuple[str, str] | None = None


# ---------------------------------------------------------------- evidence ----
def vlm_evidence(parsed: dict | None) -> dict[str, Any]:
    """Output đã parse của VLM -> bằng chứng ngữ nghĩa.

    parsed=None (sai schema) là THIẾU BẰNG CHỨNG, không phải "không có vật".
    """
    if not parsed or "hazard" not in parsed:
        return {"valid": False, "hazard": False}
    return {
        "valid": True,
        "hazard": bool(parsed.get("hazard")),
        "type": parsed.get("type"),
        "position": parsed.get("position"),
        "distance": parsed.get("distance"),   # chỉ mang tính tham khảo, không quyết
        "action": parsed.get("action", "none"),
    }


def depth_evidence(depth_map, masks=None) -> dict[str, Any]:
    """Depth map (numpy 2D) -> tín hiệu hình học SCALE-FREE của lối đi.

    near_rank = 1 - free_min/free_max trên các cột của lối đi: 1 = bị chắn ngay
    trước mặt, 0 = lối thông thoáng tới cuối khung. Không phụ thuộc thang mét.
    """
    import numpy as np

    if depth_map is None:
        return {"valid": False, "reason": "no_map"}
    band, col = masks if masks else dpath.path_masks(*depth_map.shape)
    free, grid = dpath.free_space(depth_map, band, col)
    finite = free[np.isfinite(free)]
    coverage = float((~np.isnan(grid)).sum()) / grid.size     # tỉ lệ ô đọc được
    if not len(finite):
        return {"valid": False, "reason": "no_path_pixels", "coverage": 0.0}
    shortest = float(np.min(finite)); farthest = float(np.max(finite))
    rank = float(shortest / farthest) if farthest > 0 else 0.0
    nearest_col = -1 if np.all(np.isnan(free)) else int(np.nanargmin(free))
    return {
        "valid": True,
        "free_min": round(shortest, 3),                       # đơn vị "m" của checkpoint
        "near_rank": round(max(0.0, min(1.0, 1.0 - rank)), 4),
        "col": nearest_col,                                   # 0 trái / 1 giữa / 2 phải
        "coverage": round(coverage, 3),
        "per_col_rank": [
            None if f != f or farthest <= 0 else round(1.0 - float(f) / farthest, 4)
            for f in free
        ],
    }


# ------------------------------------------------------------ cross-check -----
def position_matches(vlm_pos: str | None, geo_col: int) -> bool:
    allowed = dpath.COL_TO_POS.get(geo_col, set())
    return vlm_pos in allowed if vlm_pos else True


def map_message_code(ev_vlm: dict, action: str) -> str | None:
    """Trạng thái fused -> message_code trong cfg.ALERT_PHRASES (WAV cache §9).

    Ưu tiên: STOP (khẩn) > lớp vật cụ thể > hành động > vị trí. Không khớp code
    nào thì None — caller tự quyết fallback (TTS động hoặc bỏ qua).
    """
    obj, pos = ev_vlm.get("type"), ev_vlm.get("position") or "front"
    if action == "stop":
        return "STOP"
    type_code = {"step": "STEP_AHEAD", "vehicle": "VEHICLE_AHEAD", "person": "PERSON_AHEAD"}.get(obj)
    if type_code:
        return type_code
    act_code = {"slow": "SLOW_DOWN", "move_left": "MOVE_LEFT", "move_right": "MOVE_RIGHT"}.get(action)
    if act_code:
        return act_code
    pos_code = {"front": "OBSTACLE_FRONT", "front_left": "OBSTACLE_FRONT_LEFT",
                "left": "OBSTACLE_FRONT_LEFT", "front_right": "OBSTACLE_FRONT_RIGHT",
                "right": "OBSTACLE_FRONT_RIGHT"}.get(pos)
    return pos_code


# ---------------------------------------------------------------- decision ----
def decide_frame(ev_vlm: dict, ev_geo: dict, config: FusionConfig) -> dict[str, Any]:
    """Quyết định TRONG MỘT FRAME — chiến lược E + emergency override.

    warn_frame = VLM thấy vật AND (depth thấy chắn gần OR cửa thoát lớp mù),
    HOẶC depth khẳng định chắn CỰC GẦN (< depth_emergency_threshold) bất chấp
    VLM — lớp lỗi duy nhất còn lại sau thẩm định tay COCO (VLM nói CLEAR khi
    vật 0.3-1.7 m ngay trước mặt, 7/45 frame lỗi).
    """
    if ev_geo.get("valid") and ev_geo.get("coverage", 0) > 0:
        free_min = ev_geo.get("free_min", float("inf"))
        if free_min < config.depth_emergency_threshold:
            return {"frame_positive": True, "emergency": True, "risk": 0.95,
                    "mismatch": False, "reason": "depth_emergency"}

    if not ev_vlm.get("valid"):
        return {"frame_positive": False, "reason": "vlm_invalid"}

    vlm_hazard = ev_vlm["hazard"]
    if not vlm_hazard:
        return {"frame_positive": False, "reason": "vlm_clear"}

    obj, dist = ev_vlm.get("type"), ev_vlm.get("distance")

    if ev_geo.get("valid"):
        # Quy tắc lai đã hiệu chuẩn: ngưỡng tuyệt đối bắt vật mảnh trong phòng
        # nhỏ (pole_00: free 3.45 < 7.0), rank bắt vật chắn mềm khi mọi cột cùng
        # dừng ở tường (synth_person_front: rank 0.68 >= 0.5). Bắt cứ tín hiệu nào.
        geo_near = (ev_geo.get("coverage", 0) > 0
                    and (ev_geo.get("free_min", float("inf")) < config.free_abs_threshold
                         or ev_geo["near_rank"] >= config.near_rank_threshold))
        escape = obj in BLIND_TYPES and dist == "near"
        mismatch = not position_matches(ev_vlm.get("position"), ev_geo.get("col", -1))
        frame_positive = geo_near or escape
        risk = max(0.35 if geo_near else 0.15, 0.75 if escape else 0.0) if frame_positive else 0.1
        if mismatch and frame_positive:
            risk *= config.mismatch_risk_penalty
        reason = ("geo+escape" if escape and geo_near else
                  "escape_blind_type" if escape else
                  "geometry_blocked" if geo_near else "depth_clear")
    else:
        policy = config.depth_fail_policy
        if policy == "trust_vlm_near":
            frame_positive = dist == "near"
            reason = "depth_failed_trust_vlm_near" if frame_positive else "depth_failed_vlm_not_near"
        else:
            frame_positive = False
            reason = "depth_failed_silent"
        mismatch = False
        risk = 0.55 if frame_positive else 0.05

    return {
        "frame_positive": frame_positive,
        "risk": round(risk, 4),
        "mismatch": mismatch,
        "reason": reason,
    }


def update_session(session: FusionSession, decision: dict, ev_vlm: dict,
                   now: float | None = None) -> dict[str, Any]:
    """State machine trễ + dedup + chọn message_code + gate thuyết minh (§14).

    Chỉ phát audio khi: vừa active, đổi (type,position), hoặc severity tăng —
    không lặp lại cảnh báo trùng giữa các frame.
    """
    cfg = session.config
    now = now if now is not None else time.monotonic()

    if decision.get("frame_positive"):
        session.positive_streak += 1
        session.clear_streak = 0
    else:
        session.clear_streak += 1
        session.positive_streak = 0

    was_active = session.active
    if not session.active and session.positive_streak >= cfg.confirm_frames:
        session.active = True
    if session.active and session.clear_streak >= cfg.clear_frames:
        session.active = False

    key = None
    severity = 0
    action = "none"
    emergency = bool(decision.get("emergency"))
    # Chỉ nhận diện vật từ frame CÓ BẰNG CHỨNG (positive). Frame không positive
    # nhưng VLM vẫn nói hazard (bịa trên lối trống) thì KHÔNG được đổi key,
    # không alert — chờ clear_frames frame để state tự tắt.
    if session.active and decision.get("frame_positive") and (ev_vlm.get("valid") or emergency):
        vlm_counts = ev_vlm.get("valid") and ev_vlm.get("hazard")
        if emergency:
            # depth khẳng định chắn cực gần — STOP tuyệt đối, không nhận action
            # yếu hơn từ VLM (đã gặp thật: VLM PERSON|NEAR|NONE đè mất STOP)
            action = "stop"
            key = ((ev_vlm.get("type") or "object") if vlm_counts else "object",
                   (ev_vlm.get("position") or "front") if vlm_counts else "front")
        else:
            key = (ev_vlm.get("type") or "object", ev_vlm.get("position") or "front")
            action = ev_vlm.get("action") or "slow"
            if decision.get("mismatch"):
                action = "slow" if action == "stop" else action   # chặn STOP khi lệch hướng
        severity = ACTION_SEVERITY.get(action, 1)

    just_activated = session.active and not was_active
    key_changed = session.active and key is not None and key != session.last_key
    severity_up = session.active and key == session.last_key and severity > session.last_severity
    alert = bool(just_activated or key_changed or severity_up)

    narrate = False
    if session.active and key is not None:
        scene_changed = key != session._last_narrate_key
        if (scene_changed or now - session._last_narrate_t >= cfg.narrate_min_interval_s) \
                and now - session._last_narrate_t >= cfg.narrate_min_interval_s * 0.5:
            narrate = True
            session._last_narrate_t = now
            session._last_narrate_key = key
    elif not session.active and key is None:
        pass    # đường trống: không thuyết minh gì, đỡ nói thừa

    if key is not None:
        session.last_key = key
        session.last_severity = severity

    message_code = map_message_code(ev_vlm, action) if alert else None
    return {
        "warn": session.active,
        "alert": alert,
        "message_code": message_code,
        "narrate": narrate,
        "action": action if session.active else "none",
        "type": key[0] if key else None,
        "position": key[1] if key else None,
        "severity": severity if session.active else 0,
        "streaks": {"positive": session.positive_streak, "clear": session.clear_streak},
    }
