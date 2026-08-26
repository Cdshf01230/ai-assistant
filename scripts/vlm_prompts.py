#!/usr/bin/env python3
"""Các biến thể prompt VLM, dùng chung cho 08_ab_prompt.py và 09_eval_vlm.py.

Để một chỗ duy nhất vì hai script chấm hai thứ khác nhau trên cùng bộ prompt:
  08 = latency + số token
  09 = accuracy so với assets/frames/ground_truth.json
"""

from __future__ import annotations

FIELDS = (
    "object (PERSON, VEHICLE, POLE, STEP, DOOR, FURNITURE, HOLE, WALL, OBJECT), "
    "then where it is (FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT), "
    "then what to do (SLOW, STOP, MOVE_LEFT, MOVE_RIGHT)."
)

# v1: bản đầu tiên. "Report ONLY the most urgent hazard" đã ngầm giả định LÀ CÓ
# hazard -> đo được 0% CLEAR, báo động trên cả ảnh lối đi trống.
PIPE_V1 = (
    "You are the vision module of a mobility aid for blind pedestrians. "
    "Report ONLY the single most urgent hazard in the walking path.\n"
    "Answer with exactly one line: three uppercase fields joined by '|'.\n"
    "First field is the object: PERSON, VEHICLE, POLE, STEP, DOOR, FURNITURE, "
    "HOLE, WALL, OBJECT, or CLEAR if the path is free.\n"
    "Second field is where it is: FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, or NONE.\n"
    "Third field is what the walker should do: NONE, SLOW, STOP, MOVE_LEFT, or MOVE_RIGHT.\n"
    "Valid answers look like POLE|FRONT_RIGHT|MOVE_LEFT or VEHICLE|FRONT|STOP "
    "or CLEAR|NONE|NONE.\n"
    "Output that one line only, no other words."
)

# v2: đảo mặc định sang CLEAR. Sửa được false positive nhưng lật sang false negative.
PIPE_V2 = (
    "You are the vision module of a mobility aid for blind pedestrians.\n"
    "Decide whether anything in the walker's path needs a spoken warning RIGHT NOW.\n"
    "Most frames need NO warning. In that case answer exactly: CLEAR|NONE|NONE\n"
    "Only when something actually blocks or endangers the walking path, answer "
    "three uppercase fields joined by '|': " + FIELDS + "\n"
    "Examples: CLEAR|NONE|NONE / POLE|FRONT_RIGHT|MOVE_LEFT / VEHICLE|FRONT|STOP\n"
    "Output one line only, no other words."
)

# v3: như v2 + ngưỡng khoảng cách (§16 chống báo vật ở xa).
PIPE_V3 = (
    "You are the vision module of a mobility aid for blind pedestrians.\n"
    "Warn ONLY about things within about 3 steps ahead that are in the walking path.\n"
    "Ignore anything far away, off to the side, or already passed.\n"
    "If nothing is that close and in the way, answer exactly: CLEAR|NONE|NONE\n"
    "Otherwise answer three uppercase fields joined by '|': " + FIELDS + "\n"
    "Examples: CLEAR|NONE|NONE / POLE|FRONT_RIGHT|MOVE_LEFT / VEHICLE|FRONT|STOP\n"
    "Output one line only, no other words."
)

# v4: TRUNG TÍNH — không nói bên nào là mặc định. Để tách xem model thật sự nhìn ảnh
# hay chỉ đang lặp lại prior của prompt (v1 và v2 cho kết quả ngược nhau hoàn toàn).
PIPE_V4 = (
    "You are the vision module of a mobility aid for blind pedestrians.\n"
    "Look at the frame. Decide whether the walking path immediately ahead is blocked "
    "or free. Both answers are equally likely; judge from the image alone.\n"
    "Answer three uppercase fields joined by '|': " + FIELDS + "\n"
    "If the path immediately ahead is genuinely free, answer: CLEAR|NONE|NONE\n"
    "Examples: POLE|FRONT_RIGHT|MOVE_LEFT / VEHICLE|FRONT|STOP / CLEAR|NONE|NONE\n"
    "Output one line only, no other words."
)

# v5: NÊU VẬT TRƯỚC RỒI MỚI KẾT LUẬN. Buộc model phải trích một chi tiết từ ảnh
# trước khi chọn code, thay vì nhảy thẳng vào token đầu tiên theo prior.
# Tốn thêm ~4-6 token (ngân sách §12 còn dư tới ~20 token dưới 1000 ms).
PIPE_V5 = (
    "You are the vision module of a mobility aid for blind pedestrians.\n"
    "First name in two or three words the closest thing standing in the walking path "
    "within about 3 steps, or write nothing if the path is free.\n"
    "Then write ' => ' and three uppercase fields joined by '|': " + FIELDS + "\n"
    "If you wrote nothing, the fields must be CLEAR|NONE|NONE\n"
    "Examples:\n"
    "metal lamp post => POLE|FRONT_RIGHT|MOVE_LEFT\n"
    "parked motorbikes => VEHICLE|FRONT|STOP\n"
    "nothing => CLEAR|NONE|NONE\n"
    "Output one line only, no other words."
)

# v6: TÁCH KHOẢNG CÁCH RA THÀNH TRƯỜNG RIÊNG.
# Lý do: v4 đạt recall 100% và position 87% nhưng specificity trên frame âm khó = 0%
# — nó trả CÙNG MỘT đáp án cho cây cột ở đường chân trời và cây cột ngay trước mặt.
# Nghĩa là model biết "có vật thể, ở phía nào" nhưng không biết "gần hay xa". Bắt nó
# tự quyết định "có đáng cảnh báo không" thì quyết định đó vô nghĩa; thay vào đó chỉ
# hỏi mô tả (kèm khoảng cách) rồi để Decision Engine (§16) lọc theo NEAR/MID/FAR.
# Đây là phép thử: model có phân biệt nổi 3 mức khoảng cách hay không.
PIPE_V6 = (
    "You are the vision module of a mobility aid for blind pedestrians.\n"
    "Describe the most relevant object in or near the walking path. Do not decide "
    "whether to warn; just describe what is there.\n"
    "Answer four uppercase fields joined by '|':\n"
    "1. object: PERSON, VEHICLE, POLE, STEP, DOOR, FURNITURE, HOLE, WALL, OBJECT, "
    "or CLEAR if there is genuinely nothing.\n"
    "2. where: FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT, or NONE.\n"
    "3. how far: NEAR if within about 3 steps, MID if roughly 4 to 10 steps, "
    "FAR if further than that, NONE if the object is CLEAR.\n"
    "4. suggested action: NONE, SLOW, STOP, MOVE_LEFT, or MOVE_RIGHT.\n"
    "Examples:\n"
    "POLE|FRONT_RIGHT|NEAR|MOVE_LEFT\n"
    "PERSON|FRONT|FAR|NONE\n"
    "VEHICLE|RIGHT|MID|SLOW\n"
    "CLEAR|NONE|NONE|NONE\n"
    "Output one line only, no other words."
)

PROMPTS = {
    "pipe_v1": PIPE_V1,
    "pipe_v2": PIPE_V2,
    "pipe_v3": PIPE_V3,
    "pipe_v4": PIPE_V4,
    "pipe_v5": PIPE_V5,
    "pipe_v6": PIPE_V6,
}

# v5 sinh "<mô tả> => CODE|CODE|CODE", v6 có 4 trường -> cần nhiều token hơn.
MAX_NEW_TOKENS = {"pipe_v5": 24, "pipe_v6": 16}

# Prompt dùng schema 4 trường (có khoảng cách).
FOUR_FIELD = {"pipe_v6"}


def split_v5(text: str) -> str:
    """Lấy phần code sau ' => ' của v5; prompt khác trả về nguyên văn."""
    return text.rsplit("=>", 1)[-1] if "=>" in text else text
