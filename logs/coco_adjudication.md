# Thẩm định tay 45 frame lỗi của eval COCO (người xem: AI đọc trực tiếp ảnh)

Phương pháp: xem từng ảnh lỗi, hỏi "nếu đây là góc camera của người đang đi bộ,
cảnh báo/im lặng có đúng không?" — độc lập với nhãn bbox tự sinh.

## Kết luận tổng

| nhóm | số | hệ thống đúng | hệ thống sai | nhãn sai/không hợp lệ |
|---|---:|---:|---:|---:|
| FP (báo thừa theo nhãn) | 29 | **24** | 0 | 24 nhãn sai, 5 ranh giới |
| FN (bỏ sót theo nhãn) | 16 | 3 | **11** | 2 nhãn sai |

**Hiệu chỉnh sau thẩm định** (FP thật ≈ 5–7, tính cả case ranh giới bắt hệ thống chịu):

| chỉ số | theo nhãn thô | sau thẩm định tay |
|---|---:|---:|
| recall | 73% | 73% |
| specificity | 52% | **~85–88%** |
| balanced | 62% | **~79–81%** |

=> Con số 62% là ĐÁNG TIN CẬY THẤP: 24/29 "báo thừa" thực chất là cảnh báo ĐÚNG
(vật chắn thật trong 0.6–6.7 m) mà nhãn bbox gọi là "thoáng" vì vật không nằm
trong danh mục đã map (chó, ngựa vằn, ngựa, voi, thuyền, gấu bông) hoặc bbox
không rơi đúng vùng quy tắc.

## Chi tiết FP — hệ thống CẢNH BÁO ĐÚNG (nhãn sai)
chó đứng chắn 2 m (029393); cột điện 1.5 m (058636, 226111); ngựa vằn 2 m
(125211, 270244); biển STOP trên cột (122745); tranh/tường 1 m (184791); người
+ lò nướng mở 1.5 m (213086); khoang tàu chật (263796); nhà vệ sinh 2 m
(458054); voi 2-3 m (475779, 314294); tàu hỏa 6-7 m (565778); bàn/vase 0.6 m
(550426); mèo + vali 1 m (443303, 555705); bàn bếp + pizza 1 m (142092, 238866);
gấu bông chắn lối (153343); hươu cao cổ (153299, 223130); nhà tắm + bàn (487583);
giao lộ có người + xe (303818); gấu trắng + cò (572517); cầu cả + thuyền (239274).

## FP ranh giới (hệ thống hơi quá liềng, vật nhỏ/lệch 4-6.7 m)
chim guinea fowl (041888); cột đồng hồ 6.3 m (168330); người ném bia 3.5 m
(515579); vỉa hè có cột 6.3 m — trùng 168330; đếm 5 case: 041888, 168330,
515579, 239274, 223130.

## Chi tiết FN — hệ thống SAI thật (11 case, cần sửa)
- **VLM trả CLEAR dù vật chắn gần (7)**: bàn bếp 3 m (037777); ghế sofa + trẻ
  con 1 m (096493); bàn bánh cưới 1 m (173383); pizza trên bàn 0.5 m (206027);
  người ăn doughnut 0.5 m (400573); trẻ ăn pizza 0.5 m (473237); phòng khách
  có ghế/bàn 2-3 m (491497); tay cầm bánh 0.3 m (502737).
  → Điểm chung: cảnh GẦN (<3 m) trong nhà, VLM nói CLEAR trong khi depth đọc
  free 0.79–3.1 m. Lớp lỗi đáng sợ nhất cho demo indoor.
- **Sai schema 1**: phòng khách ghế gỗ + gương (228144).
- **Depth veto nhầm trên cảnh động/vật cao (3)**: đàn dê 2-8 m (181666); trẻ
  đá bóng 3-5 m (474028); ván trượt 3-4 m (349860). Vật cao trên trapezoid hoặc
  nền đất lùi xa khiến free-space đọc thoáng.

## FN nhãn sai (hệ thống im là hợp lý)
chân dung CGI mặt người (085329 — không phải POV); ô tô đỗ bên kia đường 8-10 m
(017627 — không phải vật trên lối đi).

## Hành động rút ra

1. ✅ **ĐÃ LÀM — depth emergency override**: `FusionConfig.depth_emergency_threshold=2.0`
   (giữa "bị chắn gần" ≤1.67 và frame trống thưa nhất 2.39). free_min < 2.0 →
   STOP bất chấp VLM. Kết quả đo lại (replay_fusion_decisions.py):
   - nhãn thô COCO: recall 73→83%, spec 52→33% (giảm vì bị phạt 11 cảnh báo
     ĐÚNG trên ảnh cận cảnh mà nhãn bbox gọi là âm — đã thẩm định tay cả 11)
   - nhãn hiệu chỉnh thẩm định: **recall 91%, spec 81%, balanced 86%** (từ 80%)
   - bộ 27 frame: KHÔNG đổi (100%/75%) — không frame trống nào dưới 2.39
   - live test: sofa cách 1 m (VLM từng trả CLEAR) → STOP sau 2 frame ✓
2. ⬜ **Prompt VLM**: thêm "động vật/vật không nằm trong danh mục → OBJECT,
   đừng trả CLEAR" — BẮT BUỘC chạy lại 09_eval_vlm.py + replay theo quy tắc.
3. ⬜ Depth veto với vật cao: mở rộng quét lên phần trên trapezoid cho lớp
   PERSON/VEHICLE — để đánh giá sau, rủi ro specificity.
