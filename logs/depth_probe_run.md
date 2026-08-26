# Depth probe and VLM fusion run

- Host: `ip-172-31-87-179`
- Started: `2026-08-24 06:54:58 UTC`

## 1. Depth probe

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/287 [00:00<?, ?it/s]Loading weights: 100%|██████████| 287/287 [00:00<00:00, 3428.35it/s]
Depth : depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
Load  : 2.2s | VRAM 48 MiB | fp16
Frames: 27 (15 có vật cản / 12 trống) | 640px | p10 | mét

--- Latency depth (27 lần) ---
  median 10 ms | min/max 10/100 ms   (VLM 4B đang là ~845 ms)

--- Quét ngưỡng, min 3 cột (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%  <-
       2.00m     0%   100%   100%    50%    0%
       2.50m     0%   100%   100%    50%    0%
       3.00m     0%   100%   100%    50%    0%
       4.00m     0%   100%   100%    50%    0%
       5.00m     0%    92%   100%    46%    0%
       6.00m    13%    92%   100%    52%  100%  <-
       8.00m    33%    83%   100%    58%   80%  <-
      10.00m    47%    58%    75%    52%   71%
      15.00m    87%    58%    75%    73%   77%  <-

--- Quét ngưỡng, chỉ cột giữa (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%  <-
       2.00m     0%   100%   100%    50%    0%
       2.50m     0%   100%   100%    50%    0%
       3.00m     0%   100%   100%    50%    0%
       4.00m     0%   100%   100%    50%    0%
       5.00m     0%    92%   100%    46%    0%
       6.00m    13%    92%   100%    52%  100%  <-
       8.00m    20%    83%   100%    52%  100%
      10.00m    33%    58%    75%    46%   80%
      15.00m    67%    58%    75%    62%   80%  <-

--- So với baseline ---
  cấu hình                  recall  specif  sp.khó  balanc
  VLM 4B v6 (near+mid)       100%    58%     0%    79%
  VLM 4B v6 (chỉ near)        87%    75%    25%    81%
  depth min-3-cột @15.0       87%    58%    75%    73%
  depth cột-giữa @15.0        67%    58%    75%    62%

--- Cùng-1-cây-cột (phép thử VLM đã trượt) ---
  synthnear_pole.jpg     min3=  5.46 giữa=  5.46 3cột=[6.0, 5.46, 6.06] gần-nhất=giữa
  synthfar_pole.jpg      min3= 25.99 giữa= 26.39 3cột=[26.45, 26.39, 25.99] gần-nhất=phải
  synthside_pole.jpg     min3=  9.66 giữa=  9.67 3cột=[9.86, 9.67, 9.66] gần-nhất=phải
  cột-xa xa hơn cột-gần?          CÓ  (25.99 vs 5.46)
  cột-lệch-bên bị cột-giữa bỏ qua? CÓ  (9.67 vs 5.46)
  => Depth phân biệt được xa/gần, thứ VLM không làm được.

--- FRAME THẬT (ngưỡng min-3-cột 15.00) ---
  CLR clear_00.jpg               min3= 24.42 giữa= 34.53 cột=trái  ok (âm khó)
  CLR clear_01.jpg               min3= 31.86 giữa= 68.38 cột=phải  ok
  HAZ clear_02.jpg               min3= 18.80 giữa= 30.15 cột=phải  BỎ SÓT
  CLR clearpath_00.jpg           min3= 53.06 giữa= 53.06 cột=giữa  ok
  HAZ clearpath_01.jpg           min3= 10.84 giữa= 12.23 cột=phải  ok
  HAZ crowd_00.jpg               min3= 10.13 giữa= 11.70 cột=trái  ok
  HAZ crowd_01.jpg               min3= 14.19 giữa= 22.66 cột=trái  ok
  HAZ door_00.jpg                min3= 19.70 giữa= 21.09 cột=trái  BỎ SÓT
  HAZ door_01.jpg                min3=  7.98 giữa= 10.56 cột=phải  ok
  HAZ lowlight_00.jpg            min3= 11.83 giữa= 24.50 cột=trái  ok
  HAZ lowlight_01.jpg            min3=  8.15 giữa=  8.80 cột=phải  ok
  HAZ pole_00.jpg                min3=  8.88 giữa=  8.88 cột=giữa  ok
  HAZ street_00.jpg              min3= 10.38 giữa= 10.38 cột=giữa  ok
  CLR street_01.jpg              min3= 17.17 giữa= 23.00 cột=trái  ok
  HAZ vehicle_00.jpg             min3=  7.80 giữa= 10.23 cột=phải  ok
  HAZ vehicle_01.jpg             min3= 14.45 giữa= 19.08 cột=trái  ok

--- FRAME SYNTHETIC (ngưỡng min-3-cột 15.00) ---
  HAZ synth_person_front.jpg     min3=  7.10 giữa=  7.10 cột=giữa  ok
  HAZ synth_pole_right.jpg       min3=  5.25 giữa=  5.33 cột=phải  ok
  CLR synthclear_dark.jpg        min3=  4.25 giữa=  4.34 cột=phải  BÁO THỪA
  CLR synthclear_grey.jpg        min3=  8.85 giữa=  8.85 cột=giữa  BÁO THỪA
  CLR synthclear_indoor.jpg      min3=  6.63 giữa=  6.63 cột=giữa  BÁO THỪA
  CLR synthclear_tiled.jpg       min3= 27.56 giữa= 27.74 cột=phải  ok
  CLR synthclear_warm.jpg        min3=  9.13 giữa=  9.13 cột=giữa  BÁO THỪA
  CLR synthfar_person.jpg        min3= 19.27 giữa= 19.27 cột=giữa  ok (âm khó)
  CLR synthfar_pole.jpg          min3= 25.99 giữa= 26.39 cột=phải  ok (âm khó)
  HAZ synthnear_pole.jpg         min3=  5.46 giữa=  5.46 cột=giữa  ok
  CLR synthside_pole.jpg         min3=  9.66 giữa=  9.67 cột=phải  BÁO THỪA (âm khó)
Kết quả thô -> logs/depth_outdoor.json

```

- Exit code: `0`

## 2. VLM-depth fusion

```text
VLM   : Qwen/Qwen3-VL-4B-Instruct / pipe_v6
Depth : depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf | free3 < 15.00m | p10 | 10 ms
Frames: 27 khớp cả hai dump

chiến lược          recall  specif  sp.khó   balanc   type  hướng
A. VLM near+mid      100%    58%     0%     79%   80%   80%
B. VLM chỉ near       87%    75%    25%     81%   85%   85%
C. chỉ depth          87%    58%    75%     73%    0%   77%
D. VLM AND depth      87%    92%    75%     89%   77%   77%

A. VLM near+mid
   bỏ sót  (0): —
   báo thừa(5): clear_00.jpg (âm khó), clear_01.jpg, synthfar_pole.jpg (âm khó), synthfar_person.jpg (âm khó), synthside_pole.jpg (âm khó)
B. VLM chỉ near
   bỏ sót  (2): clear_02.jpg, vehicle_00.jpg
   báo thừa(3): clear_00.jpg (âm khó), synthfar_pole.jpg (âm khó), synthside_pole.jpg (âm khó)
C. chỉ depth
   bỏ sót  (2): clear_02.jpg, door_00.jpg
   báo thừa(5): synthclear_grey.jpg, synthclear_warm.jpg, synthclear_dark.jpg, synthclear_indoor.jpg, synthside_pole.jpg (âm khó)
D. VLM AND depth
   bỏ sót  (2): clear_02.jpg, door_00.jpg
   báo thừa(1): synthside_pole.jpg (âm khó)

=== Kết luận ===
  recall      100% -> 87%
  specificity 58% -> 92%
  âm khó      0% -> 75%   <- đây là chỗ VLM đơn độc bằng 0
  balanced    79% -> 89%
  ⚠️ Ghép depth làm recall TỤT (100% -> 87%): mất 2 frame có vật cản. Với thiết bị dẫn đường thì bỏ sót đắt hơn báo thừa — phải nới ngưỡng hoặc coi depth=nan là 'gần'.

LƯU Ý: 12 frame âm — specificity vẫn nhảy từng bước lớn. §8 vẫn cần 100-500 frame thật.

```

- Exit code: `0`
- Finished: `2026-08-24 06:55:08 UTC`
