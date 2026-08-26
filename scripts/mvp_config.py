#!/usr/bin/env python3
"""Cấu hình dùng chung cho mọi script model."""

from __future__ import annotations

import os

MODELS_DIR = "/home/ubuntu/ai-assistant/models"
os.environ.setdefault("HF_HOME", f"{MODELS_DIR}/hf")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
# Tokenizer parallelism gây warning khi fork trong uvicorn worker
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

STT_MODEL = "vinai/PhoWhisper-medium"
STT_MODEL_BACKUP = "vinai/PhoWhisper-small"
# ĐÃ ĐO (09_eval_vlm.py, 27 frame có nhãn): 4B gọi đúng tên vật 85% so với 33% của
# 2B, balanced accuracy 81% so với 75%. Trả giá bằng latency 845 ms so với 468 ms
# và VRAM 8.5 GiB so với 4.2 GiB. Chọn 4B vì câu cảnh báo sai tên thì vô dụng.
VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VLM_MODEL_2B = "Qwen/Qwen3-VL-2B-Instruct"   # nhanh hơn, kém chính xác hơn
VLM_MODEL_BACKUP = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
TTS_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
TTS_VOICE = "Phạm Tuyên"
# Checkpoint depth đang dùng trong api_main.py — NGUỒN DUY NHẤT của model id này.
# Đã hiệu chuẩn free_abs_threshold=7.0 trên chính nó (FusionConfig trong fusion.py).
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

# ĐÃ ĐO: PhoWhisper trên CPU 2710 ms, trên GPU 165 ms (chênh 16x). §24 dự kiến STT
# ở CPU nhưng số đo nói ngược — model chỉ ~0.5 GiB, T4 còn dư ~6 GiB sau VLM 4B, và
# STT là push-to-talk nên không chạy song song liên tục với luồng vision.
STT_DEVICE = "cuda"
# TTS thì đúng như §24: đường CPU/ONNX int8 torch-free, RTF 0.61, không cần GPU.
TTS_DEVICE = "cpu"

# T4 = sm_75: không có bf16 native, không có FlashAttention-2.
GPU_DTYPE = "float16"
ATTN_IMPL = "sdpa"

# §13: không stream 1080p. Resize về cạnh dài ~640 rồi JPEG q80.
# Đã đo (07_tune_vlm.py): hạ 640 -> 448 chỉ tiết kiệm ~78 ms TTFT vì decode mới
# là bottleneck. Nên giữ 640 để VLM "nhìn" rõ hơn.
FRAME_LONG_EDGE = 640
JPEG_QUALITY = 80

# --- Prompt VLM ---------------------------------------------------------------
# §7: ép VLM trả schema ngắn, cố định.
#
# LATENCY — đã đo trên T4 (07_tune_vlm.py, 640px, fp16):
#   JSON schema  -> 23.6 token, 1103 ms   (§12: CHƯA ĐẠT)
#   pipe-compact ->  8.0 token,  468 ms   (§12: tốt)
# Decode trên T4 tốn ~33-40 ms/token nên SỐ TOKEN SINH RA là biến quan trọng nhất
# của latency, không phải resolution. => Production dùng PIPE.
#
# ACCURACY — đã đo trên 27 frame có nhãn tay (09_eval_vlm.py, xem
# assets/frames/ground_truth.json và logs/eval_vlm_*.json):
#   model prompt              recall  spec  spec.âm-khó  balanced  đúng-tên-vật
#   2B    v1 gốc               100%    8%       0%          54%        47%
#   2B    v3 ngưỡng-3-bước      27%   83%      50%          55%        75%
#   2B    v4 trung tính        100%   50%       0%          75%        33%
#   4B    v4 trung tính         53%   67%      25%          60%        75%
#   4B    v6 + khoảng cách      87%   75%      25%          81%        85%  <- dùng
#   4B    v6, nhận cả MID      100%   58%       0%          79%        85%  <- ngưỡng
#
# Hai bài học đã trả giá để biết:
#
# 1. Prompt lệch thì model lệch theo. "Report ONLY the most urgent hazard" (v1) ngầm
#    khẳng định LÀ CÓ hazard -> báo động trên 25/27 frame kể cả ảnh trống. "Most
#    frames need no warning" (v3) khẳng định ngược -> bỏ sót 11/15 vật cản thật.
#    Prompt phải TRUNG TÍNH. Sửa prompt thì BẮT BUỘC chạy lại 09_eval_vlm.py.
#
# 2. Model KHÔNG suy luận khoảng cách, và lên 4B cũng không sửa được.
#    Ba frame synthetic dựng từ đúng một cây cột, chỉ đổi kích thước/vị trí:
#      cột ngay trước mặt   -> pole|front|near|move_left
#      đúng cột đó, ở xa    -> pole|front|near|move_left   (y nguyên)
#    Trên 27 frame model trả near 16 lần, mid 4 lần, far 0 lần — nhãn far không bao
#    giờ được dùng. => Trường khoảng cách dưới đây gần như không mang thông tin;
#    §16 KHÔNG lọc được cảnh báo bằng nó. Phải có tín hiệu hình học riêng
#    (depth monocular, hoặc suy từ chiều cao camera + cạnh dưới bbox). Xem README.
VLM_PROMPT = VLM_PROMPT_PIPE = (
    "You are the vision module of a mobility aid for blind pedestrians.\n"
    "Describe the most relevant object in or near the walking path. Do not decide "
    "whether to warn; just describe what is there.\n"
    "Answer four uppercase fields joined by '|':\n"
    "1. object: PERSON, ANIMAL, VEHICLE, POLE, STEP, DOOR, FURNITURE, HOLE, WALL, OBJECT, "
    "or CLEAR if there is genuinely nothing.\n"
    "   Use ANIMAL for any animal (dog, horse, elephant, bird, etc.). "
    "   Use OBJECT for any object not listed above. "
    "   Do NOT answer CLEAR if there is any object or animal in the path.\n"
    "2. where: FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, or NONE.\n"
    "3. how far: NEAR if within about 3 steps, MID if roughly 4 to 10 steps, "
    "FAR if further than that, NONE if the object is CLEAR.\n"
    "4. suggested action: NONE, SLOW, STOP, MOVE_LEFT, or MOVE_RIGHT.\n"
    "Examples:\n"
    "POLE|FRONT_RIGHT|NEAR|MOVE_LEFT\n"
    "PERSON|FRONT|FAR|NONE\n"
    "VEHICLE|RIGHT|MID|SLOW\n"
    "ANIMAL|FRONT|NEAR|STOP\n"
    "OBJECT|FRONT_LEFT|MID|SLOW\n"
    "CLEAR|NONE|NONE|NONE\n"
    "Output one line only, no other words."
)
VLM_MAX_NEW_TOKENS = 16  # schema 4 trường sinh ~10 token; 16 là đủ dư

# Ngưỡng để Decision Engine quyết có phát cảnh báo hay không.
# Chọn {near, mid} thay vì {near}: recall 100% so với 87%, đổi lấy specificity
# 58% so với 75%. Với thiết bị dẫn đường thì bỏ sót vật cản nguy hiểm hơn nói thừa
# một câu, và §14 (không lặp lại cảnh báo trùng) còn lọc bớt được phần nói thừa.
# Nhưng phải nói rõ: specificity 58% = ~4/10 frame trống vẫn bị cảnh báo. Chưa
# đủ tốt để giao cho người dùng thật, xem phần §30 trong README.
VLM_WARN_DISTANCES = frozenset({"near", "mid"})


def should_warn(parsed: dict | None) -> bool:
    """Decision Engine (§16): có phát cảnh báo cho frame này hay không.

    Quyết định nằm ở ĐÂY, không nằm ở VLM — đã đo được rằng khi bắt VLM tự quyết
    "có đáng cảnh báo không" thì nó chỉ lặp lại giả định của prompt. VLM chỉ mô tả.

    parsed=None (sai schema) -> im lặng, tuyệt đối không đoán.
    """
    if parsed is None or not parsed.get("hazard"):
        return False
    dist = parsed.get("distance")
    if dist is None:          # schema 3 trường: không có gì để lọc
        return True
    return dist in VLM_WARN_DISTANCES

# Giữ lại bản JSON để benchmark đối chứng — KHÔNG dùng cho production vì chậm 2.5x.
VLM_PROMPT_JSON = (
    "You are the vision module of a mobility aid for blind pedestrians. "
    "Look at the camera frame and report ONLY the single most urgent hazard "
    "in the walking path. Reply with ONE line of compact JSON, no markdown, "
    "no explanation, using exactly this schema:\n"
    '{"hazard":bool,"type":str,"position":str,"severity":int,"action":str}\n'
    "type: one of person|vehicle|pole|step|door|furniture|hole|wall|object|none\n"
    "position: one of front|front_left|front_right|left|right|none\n"
    "severity: 0=none 1=notice 2=warning 3=critical\n"
    "action: one of none|slow|stop|move_left|move_right\n"
    "If the path is clear reply exactly: "
    '{"hazard":false,"type":"none","position":"none","severity":0,"action":"none"}'
)
# Tương thích ngược với script cũ
VLM_SYSTEM_PROMPT = VLM_PROMPT

# Vocabulary hợp lệ để validate output — token lạ nào cũng coi là hallucination.
VLM_TYPES = frozenset(
    "PERSON ANIMAL VEHICLE POLE STEP DOOR FURNITURE HOLE WALL OBJECT CLEAR".split()
)
VLM_POSITIONS = frozenset("FRONT FRONT_LEFT FRONT_RIGHT LEFT RIGHT NONE".split())
VLM_ACTIONS = frozenset("NONE SLOW STOP MOVE_LEFT MOVE_RIGHT".split())

# Trường thứ 4 (khoảng cách). ĐÃ ĐO: schema 3 trường cho recall 100% nhưng
# specificity trên frame "vật ở xa / lệch bên" = 0% — model trả y nguyên một đáp án
# cho cây cột ở đường chân trời và cây cột ngay mặt. Không có trường khoảng cách thì
# Decision Engine (§16) không có gì để lọc.
VLM_DISTANCES = frozenset("NEAR MID FAR NONE".split())


def parse_pipe4(text: str) -> dict | None:
    """Parse 'POLE|FRONT|NEAR|MOVE_LEFT' -> dict. None nếu sai schema.

    Nhận cả schema 3 trường (khi đó distance=None) để tương thích ngược.
    """
    line = text.strip().splitlines()[0].strip().strip("`\"' ") if text.strip() else ""
    parts = [p.strip().upper() for p in line.split("|")]
    if len(parts) == 3:
        return parse_pipe(text)
    if len(parts) != 4:
        return None
    obj, pos, dist, act = parts
    if (obj not in VLM_TYPES or pos not in VLM_POSITIONS
            or dist not in VLM_DISTANCES or act not in VLM_ACTIONS):
        return None
    return {
        "hazard": obj != "CLEAR",
        "type": obj.lower(),
        "position": pos.lower(),
        "distance": dist.lower(),
        "action": act.lower(),
    }


def parse_pipe(text: str) -> dict | None:
    """Parse 'PERSON|FRONT_RIGHT|MOVE_LEFT' -> dict, hoặc None nếu sai schema.

    Trả None thì Decision Engine phải bỏ qua frame đó, KHÔNG được đoán —
    thà im lặng một frame hơn là cảnh báo sai (§16).
    """
    line = text.strip().splitlines()[0].strip().strip("`\"' ") if text.strip() else ""
    parts = [p.strip().upper() for p in line.split("|")]
    if len(parts) != 3:
        return None
    obj, pos, act = parts
    if obj not in VLM_TYPES or pos not in VLM_POSITIONS or act not in VLM_ACTIONS:
        return None
    return {
        "hazard": obj != "CLEAR",
        "type": obj.lower(),
        "position": pos.lower(),
        "action": act.lower(),
    }

# §9 + §17.3: các câu cảnh báo tối quan trọng -> phát WAV có sẵn, không inference TTS.
ALERT_PHRASES: dict[str, str] = {
    "STOP": "Dừng lại.",
    "TURN_LEFT": "Rẽ trái.",
    "TURN_RIGHT": "Rẽ phải.",
    "OBSTACLE_FRONT": "Có vật cản phía trước.",
    "OBSTACLE_FRONT_LEFT": "Có vật cản phía trước bên trái.",
    "OBSTACLE_FRONT_RIGHT": "Có vật cản phía trước bên phải.",
    "MOVE_LEFT": "Đi lệch sang trái.",
    "MOVE_RIGHT": "Đi lệch sang phải.",
    "SLOW_DOWN": "Đi chậm lại.",
    "STEP_AHEAD": "Có bậc thang phía trước.",
    "VEHICLE_AHEAD": "Có xe phía trước.",
    "PERSON_AHEAD": "Có người phía trước.",
    "PATH_CLEAR": "Đường thông thoáng.",
    "ARRIVED": "Đã đến nơi.",
    "REPEAT_PLEASE": "Xin nói lại.",
    "GPS_LOST": "Mất tín hiệu định vị.",
}

ALERT_AUDIO_DIR = "/home/ubuntu/ai-assistant/assets/audio"
