# Golden test cases — hồi quy fusion không cần GPU

Bộ test đóng gói **ảnh + nhãn + đầu vào đã đo + kỳ vọng hành vi** để bất kỳ thay
đổi nào của `scripts/fusion.py` / `scripts/depth_path.py` / `scripts/mvp_config.py`
đều bị kiểm tra tức thì mà KHÔNG cần GPU hay model.

## Chạy

```bash
cd /home/ubuntu/ai-assistant
.venv/bin/python tests/run_golden_tests.py        # exit 0 = PASS
.venv/bin/python tests/run_golden_tests.py -v     # in chi tiết case lệch
```

## Bộ case

| file | nội dung | nguồn |
|---|---|---|
| `golden/cases_core27.json` | 27 frame chụp/thẩm định tay (15 dương, 8 âm thường, 4 âm khó) | `assets/frames/ground_truth.json`, output VLM thật `logs/eval_vlm_qwen3vl4b.json`, dump depth `logs/depth_indoor_large.json` |
| `golden/cases_coco120.json` | 120 ảnh COCO val2017, nhãn bbox thô + hiệu chỉnh theo thẩm định tay | `logs/vlm_public_coco.json`, `logs/coco_adjudication.md` |
| `golden/manifest.json` | SHA-256 của 151 ảnh nguồn | — |

Mỗi case: `{image, label, inputs:{vlm_raw, vlm_parsed, geo}, expect}` trong đó
`inputs` là số đo THẬT của pipeline (không mô phỏng), `expect` ghim hành vi của
fusion tại thời điểm đóng gói.

## Ba mức kiểm tra

- **A. Invariant an toàn** — dương thật phải cảnh báo (recall 100% trên core27);
  frame có `free_min < depth_emergency_threshold` phải STOP kèm reason
  `depth_emergency` bất chấp VLM; depth hỏng thì theo `depth_fail_policy`.
- **B. Regression fingerprint** — từng case so `frame_positive/reason/risk/mismatch`
  với bản ghi. Lệch nghĩa là logic fusion đã đổi hành vi: thẩm định xem đổi đúng
  chủ đích không, rồi mới chạy lại builder để cập nhật golden.
- **C. Aggregate** — core27: recall ≥100%, spec ≥75%; coco120 (nhãn hiệu chỉnh):
  recall ≥90%, spec ≥80%, fp ≤5 (đúng 5 case ranh giới đã thẩm định).
  Số chuẩn: tp/fp/fn/tn = 85/5/8/22 — khớp `benchmarks/replay_fusion_decisions.py`.

Kèm 6 unit hành vi: depth-hỏng theo policy, cửa thoát step/hole/door, cross-check
hướng hạ risk, emergency override + ưu tiên STOP, state machine trễ/dedup/đổi-vật,
parse schema từ chối token lạ.

## Cập nhật golden khi nào?

```bash
.venv/bin/python tests/build_golden_cases.py
```

CHỈ khi thay đổi NGUỒN: thêm frame mới có nhãn, chạy lại VLM/depth dump, hoặc
thẩm định tay bổ sung. KHÔNG dùng builder để "tẩy" test fail sau khi sửa fusion —
trước khi build lại phải trả lời được: hành vi mới đúng vì số đo nào?
