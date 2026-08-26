#!/usr/bin/env bash
# Vòng 2 của depth probe. Vòng 1 đã trả lời được câu hỏi chính (depth CÓ cấp được
# tín hiệu khoảng cách mà VLM mù), nhưng để lại 3 chỗ phải đóng:
#
#   1. Ngưỡng chọn theo balanced accuracy -> đổi mất recall 100% -> 87%.
#      Đã sửa: 11 và 12 giờ chọn theo an toàn (giữ recall 100% trước).
#   2. Hai frame bị bỏ sót (door_00 bậc thang, clear_02 người nằm) đều là vật
#      THẤP/PHẲNG — không chắn tia nhìn nên depth đọc ra khoảng trống phía sau.
#      Đã thêm chiến lược E: depth không được phủ quyết lớp step/hole/door.
#   3. Thang "mét" lệch 4-5 lần vì checkpoint train trên VKITTI (camera gắn xe).
#      Thử checkpoint indoor (Hypersim, 0-20m) xem có sát tầm người hơn không.
#
# Chạy:  bash scripts/run_depth_round2.sh
# Log:   logs/depth_round2.md

set -u
cd /home/ubuntu/ai-assistant || exit 1
PY=.venv/bin/python
OUT=logs/depth_round2.md
export HF_HOME=/home/ubuntu/ai-assistant/models/hf

: > "$OUT"
{
  echo "# Depth probe vòng 2"
  echo
  echo "- Host: \`$(hostname)\`"
  echo "- Started: \`$(date -u '+%Y-%m-%d %H:%M:%S UTC')\`"
} >> "$OUT"

step() {   # step <tiêu đề> <lệnh...>
  local title="$1"; shift
  echo "" >> "$OUT"
  echo "## $title" >> "$OUT"
  echo "" >> "$OUT"
  echo '```text' >> "$OUT"
  echo "\$ $*" >> "$OUT"
  "$@" >> "$OUT" 2>&1
  local rc=$?
  echo '```' >> "$OUT"
  echo "" >> "$OUT"
  echo "- Exit code: \`$rc\`" >> "$OUT"
  printf '%-46s rc=%s\n' "$title" "$rc"
}

# 1. Ghép lại trên dump CŨ — không cần GPU, xác nhận ngay chiến lược E và ngưỡng.
step "Ghép VLM+depth, quét ngưỡng, thêm chiến lược E" \
  $PY scripts/12_fuse_vlm_depth.py \
      logs/eval_vlm_qwen3vl4b.json logs/depth_outdoor.json

# 2. Chạy lại probe với quy tắc chọn ngưỡng đã sửa + quét mịn 14-26.
step "Depth outdoor, quy tắc chọn ngưỡng mới" \
  $PY scripts/11_depth_probe.py --dump logs/depth_outdoor.json

# 3. Vật mảnh (cột đèn) dễ lọt percentile 10 -> thử p3.
step "Depth outdoor, p3 (bắt vật mảnh)" \
  $PY scripts/11_depth_probe.py --pct 3 --dump logs/depth_outdoor_p3.json

# 4. Checkpoint indoor: Hypersim 0-20m, gần tầm người đi bộ hơn VKITTI 0-80m.
#    Tải trực tiếp, bỏ qua proxy (proxy đo được 1.8 MB/s vs 83 MB/s trực tiếp).
step "Tải checkpoint depth indoor" \
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  $PY scripts/01_download_models.py --only depth_indoor

step "Depth indoor (thang có sát tầm người hơn?)" \
  $PY scripts/11_depth_probe.py --which indoor --dump logs/depth_indoor.json

step "Ghép VLM + depth indoor" \
  $PY scripts/12_fuse_vlm_depth.py \
      logs/eval_vlm_qwen3vl4b.json logs/depth_indoor.json

# 5. 08 đã sửa (bỏ đường chấm điểm theo CLEAR-rate vì nó từng chọn ra prompt bỏ
#    sót 11/15 vật cản) nhưng CHƯA chạy lần nào — chỉ mới compile. Chạy để xác nhận.
step "08_ab_prompt.py (chỉ đo tốc độ, xác nhận bản sửa)" \
  $PY scripts/08_ab_prompt.py

echo "" >> "$OUT"
echo "- Finished: \`$(date -u '+%Y-%m-%d %H:%M:%S UTC')\`" >> "$OUT"
echo
echo "Log -> $OUT"
