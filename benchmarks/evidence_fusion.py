#!/usr/bin/env python3
"""Generic VLM/depth evidence fusion with temporal hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FusionConfig:
    warn_threshold: float = 0.60
    stop_threshold: float = 0.82
    clear_threshold: float = 0.35
    confirm_frames: int = 2
    clear_frames: int = 3


@dataclass
class TrackState:
    active: bool = False
    positive_streak: int = 0
    clear_streak: int = 0


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def depth_evidence(features: dict[str, Any]) -> float:
    """Return geometry evidence; NaN/invalid depth is uncertainty, not clearance."""
    valid = clamp(float(features.get("valid_pixel_ratio", 0.0)))
    obstacle = clamp(float(features.get("obstacle_score", 0.0)))
    contrast = clamp(float(features.get("depth_contrast_score", 0.0)))
    free_space = features.get("free_space_score")
    free_signal = 0.0 if free_space is None else 1.0 - clamp(float(free_space))
    quality = 0.5 + 0.5 * valid
    return clamp(quality * (0.45 * obstacle + 0.35 * contrast + 0.20 * free_signal))


def fuse_evidence(
    vlm: dict[str, Any] | None,
    depth: dict[str, Any] | None,
    state: TrackState | None = None,
    config: FusionConfig = FusionConfig(),
) -> dict[str, Any]:
    """Fuse semantic and geometric evidence without class-specific exceptions."""
    state = state or TrackState()
    if not vlm or not vlm.get("hazard"):
        vlm_score = 0.0
    else:
        vlm_score = clamp(float(vlm.get("confidence", 0.70)))
    geometry_score = depth_evidence(depth or {}) if depth else 0.0
    depth_valid = bool(depth and float(depth.get("valid_pixel_ratio", 0.0)) > 0.0)
    risk = clamp(0.55 * vlm_score + 0.45 * geometry_score)

    if risk >= config.warn_threshold:
        state.positive_streak += 1
        state.clear_streak = 0
    elif risk <= config.clear_threshold:
        state.clear_streak += 1
        state.positive_streak = 0
    else:
        state.positive_streak = 0
        state.clear_streak = 0
    if not state.active and state.positive_streak >= config.confirm_frames:
        state.active = True
    if state.active and state.clear_streak >= config.clear_frames:
        state.active = False

    action = "NONE"
    if state.active:
        action = "STOP" if risk >= config.stop_threshold else str(vlm.get("action", "SLOW")).upper()
    return {
        "warn": state.active,
        "risk": round(risk, 4),
        "action": action,
        "type": vlm.get("type") if vlm else None,
        "position": vlm.get("position") if vlm else None,
        "depth_valid": depth_valid,
        "evidence": {
            "vlm": round(vlm_score, 4),
            "depth": round(geometry_score, 4),
        },
    }
