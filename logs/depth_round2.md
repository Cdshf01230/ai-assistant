# Depth probe vòng 2

- Host: `ip-172-31-87-179`
- Started: `2026-08-24 07:25:49 UTC`

## Ghép VLM+depth, quét ngưỡng, thêm chiến lược E

```text
$ .venv/bin/python scripts/12_fuse_vlm_depth.py logs/eval_vlm_qwen3vl4b.json logs/depth_outdoor.json
VLM   : Qwen/Qwen3-VL-4B-Instruct / pipe_v6
Depth : depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf | free3 | p10 | 10 ms
Frames: 27 khớp cả hai dump

--- Quét ngưỡng ghép (free3) ---
    ngưỡng  D recall  D spec  D khó  E recall  E spec  E khó
     8.00      33%    100%   100%      40%    100%   100%
    10.00      47%     92%    75%      53%     92%    75%
    12.00      73%     92%    75%      80%     92%    75%
    14.00      73%     92%    75%      80%     92%    75%
    15.00      87%     92%    75%      93%     92%    75%
    16.00      87%     92%    75%      93%     92%    75%
    18.00      87%     92%    75%      93%     92%    75%
    19.00      93%     92%    75%     100%     92%    75%
    19.50      93%     83%    50%     100%     83%    50%
    20.00     100%     83%    50%     100%     83%    50%
    21.00     100%     83%    50%     100%     83%    50%
    22.00     100%     83%    50%     100%     83%    50%
    24.00     100%     83%    50%     100%     83%    50%
    26.00     100%     67%     0%     100%     67%     0%
  quy tắc chọn: E giữ recall 100% ở ngưỡng thấp nhất
  => ngưỡng 19.00m (recall 100%, specificity 92%, âm khó 75%)
  dư địa tới vật cản xa nhất (19.70): -0.70 = -3.7%   <- QUÁ SÁT, ngưỡng đang bị ghim bởi 1 frame

Ngưỡng dùng: free3 < 19.00m

chiến lược          recall  specif  sp.khó   balanc   type  hướng
A. VLM near+mid      100%    58%     0%     79%   80%   80%
B. VLM chỉ near       87%    75%    25%     81%   85%   85%
C. chỉ depth          93%    50%    75%     72%    0%   71%
D. VLM AND depth      93%    92%    75%     92%   79%   79%
E. D + cửa thoát     100%    92%    75%     96%   80%   80%

A. VLM near+mid
   bỏ sót  (0): —
   báo thừa(5): clear_00.jpg (âm khó), clear_01.jpg, synthfar_pole.jpg (âm khó), synthfar_person.jpg (âm khó), synthside_pole.jpg (âm khó)
B. VLM chỉ near
   bỏ sót  (2): clear_02.jpg, vehicle_00.jpg
   báo thừa(3): clear_00.jpg (âm khó), synthfar_pole.jpg (âm khó), synthside_pole.jpg (âm khó)
C. chỉ depth
   bỏ sót  (1): door_00.jpg
   báo thừa(6): street_01.jpg, synthclear_grey.jpg, synthclear_warm.jpg, synthclear_dark.jpg, synthclear_indoor.jpg, synthside_pole.jpg (âm khó)
D. VLM AND depth
   bỏ sót  (1): door_00.jpg
   báo thừa(1): synthside_pole.jpg (âm khó)
E. D + cửa thoát
   bỏ sót  (0): —
   báo thừa(1): synthside_pole.jpg (âm khó)

=== Kết luận ===
                  recall  specif  sp.khó  balanc
  A baseline       100%    58%     0%    79%
  D AND thuần       93%    92%    75%    92%
  E đề xuất        100%    92%    75%    96%
  âm khó: 0% -> 75%   <- đây là chỗ VLM đơn độc bằng 0
  => Recall giữ nguyên 100%, specificity 58% -> 92%, âm khó 0% -> 75%.
     Không đổi gì lấy gì: depth cấp đúng tín hiệu VLM đang thiếu.
  Cửa thoát cho door/hole/step cứu được 1 frame mà AND thuần bỏ sót, giá: không mất specificity.

LƯU Ý: 12 frame âm — specificity vẫn nhảy từng bước lớn. §8 vẫn cần 100-500 frame thật.
LƯU Ý: ngưỡng trên KHÔNG phải mét thật (checkpoint VKITTI, camera gắn xe) — chỉ
       dùng được thứ tự, và phải hiệu chuẩn lại cho từng máy.
```

- Exit code: `0`

## Depth outdoor, quy tắc chọn ngưỡng mới

```text
$ .venv/bin/python scripts/11_depth_probe.py --dump logs/depth_outdoor.json
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/287 [00:00<?, ?it/s]Loading weights: 100%|██████████| 287/287 [00:00<00:00, 3568.29it/s]
Depth : depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
Load  : 0.5s | VRAM 48 MiB | fp16
Frames: 27 (15 có vật cản / 12 trống) | 640px | p10 | mét

--- Latency depth (27 lần) ---
  median 11 ms | min/max 10/90 ms   (VLM 4B đang là ~845 ms)

--- Quét ngưỡng, min 3 cột (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%
       2.00m     0%   100%   100%    50%    0%
       3.00m     0%   100%   100%    50%    0%
       4.00m     0%   100%   100%    50%    0%
       5.00m     0%    92%   100%    46%    0%
       6.00m    13%    92%   100%    52%  100%
       8.00m    33%    83%   100%    58%   80%
      10.00m    47%    58%    75%    52%   71%
      12.00m    73%    58%    75%    66%   82%
      14.00m    73%    58%    75%    66%   82%
      15.00m    87%    58%    75%    73%   77%
      16.00m    87%    58%    75%    73%   77%
      18.00m    87%    50%    75%    68%   77%
      19.00m    93%    50%    75%    72%   71%
      19.50m    93%    42%    50%    68%   71%
      20.00m   100%    42%    50%    71%   73%  <- chọn
      21.00m   100%    42%    50%    71%   73%
      22.00m   100%    42%    50%    71%   73%
      24.00m   100%    42%    50%    71%   73%
      26.00m   100%    25%     0%    62%   73%
  quy tắc chọn: recall 100% + specificity cao nhất
  dư địa tới vật cản xa nhất (19.70): +0.30 = 1.5%   <- QUÁ SÁT, ngưỡng bị ghim bởi 1 frame

--- Quét ngưỡng, chỉ cột giữa (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%
       2.00m     0%   100%   100%    50%    0%
       3.00m     0%   100%   100%    50%    0%
       4.00m     0%   100%   100%    50%    0%
       5.00m     0%    92%   100%    46%    0%
       6.00m    13%    92%   100%    52%  100%
       8.00m    20%    83%   100%    52%  100%
      10.00m    33%    58%    75%    46%   80%
      12.00m    60%    58%    75%    59%   78%
      14.00m    67%    58%    75%    62%   80%
      15.00m    67%    58%    75%    62%   80%
      16.00m    67%    58%    75%    62%   80%
      18.00m    67%    58%    75%    62%   80%
      19.00m    67%    58%    75%    62%   80%
      19.50m    73%    50%    50%    62%   82%
      20.00m    73%    50%    50%    62%   82%
      21.00m    73%    50%    50%    62%   82%
      22.00m    80%    50%    50%    65%   83%
      24.00m    87%    42%    50%    64%   77%
      26.00m    93%    42%    50%    68%   79%  <- chọn
  quy tắc chọn: KHÔNG ngưỡng nào đạt recall 100% -> đành lấy balanced cao nhất

--- Thang đo này có thật là mét không? ---
  frame có vật cản :   5.25 ..  19.70
  frame lối trống  :   4.25 ..  53.06
  Vật cản gần nhất trong bộ frame cách chừng 1-2 bước, nhưng đọc ra 5.3 'mét'.
  Checkpoint metric train trên VKITTI (camera gắn trên xe, tầm 0-80m), nên camera
  cầm tay ở tầm người là out-of-distribution: sai số thang chừng 4-5 lần. Chỉ THỨ TỰ
  dùng được. §16 KHÔNG ngưỡng được theo '3 bước chân' — ngưỡng phải hiệu chuẩn
  cho từng độ cao/góc chúc camera và sẽ trôi khi đổi cấu hình.

--- So với baseline ---
  cấu hình                  recall  specif  sp.khó  balanc
  VLM 4B v6 (near+mid)       100%    58%     0%    79%
  VLM 4B v6 (chỉ near)        87%    75%    25%    81%
  depth min-3-cột @20.0      100%    42%    50%    71%
  depth cột-giữa @26.0        93%    42%    50%    68%

--- Cùng-1-cây-cột (phép thử VLM đã trượt) ---
  synthnear_pole.jpg     min3=  5.46 giữa=  5.46 3cột=[6.0, 5.46, 6.06] gần-nhất=giữa
  synthfar_pole.jpg      min3= 25.99 giữa= 26.39 3cột=[26.45, 26.39, 25.99] gần-nhất=phải
  synthside_pole.jpg     min3=  9.66 giữa=  9.67 3cột=[9.86, 9.67, 9.66] gần-nhất=phải
  cột-xa xa hơn cột-gần?          CÓ  (25.99 vs 5.46)
  cột-lệch-bên bị cột-giữa bỏ qua? CÓ  (9.67 vs 5.46)
  => Depth phân biệt được xa/gần, thứ VLM không làm được.

--- FRAME THẬT (ngưỡng min-3-cột 20.00) ---
  CLR clear_00.jpg               min3= 24.42 giữa= 34.53 cột=trái  ok (âm khó)
  CLR clear_01.jpg               min3= 31.86 giữa= 68.38 cột=phải  ok
  HAZ clear_02.jpg               min3= 18.80 giữa= 30.15 cột=phải  ok
  CLR clearpath_00.jpg           min3= 53.06 giữa= 53.06 cột=giữa  ok
  HAZ clearpath_01.jpg           min3= 10.84 giữa= 12.23 cột=phải  ok
  HAZ crowd_00.jpg               min3= 10.13 giữa= 11.70 cột=trái  ok
  HAZ crowd_01.jpg               min3= 14.19 giữa= 22.66 cột=trái  ok
  HAZ door_00.jpg                min3= 19.70 giữa= 21.09 cột=trái  ok
  HAZ door_01.jpg                min3=  7.98 giữa= 10.56 cột=phải  ok
  HAZ lowlight_00.jpg            min3= 11.83 giữa= 24.50 cột=trái  ok
  HAZ lowlight_01.jpg            min3=  8.15 giữa=  8.80 cột=phải  ok
  HAZ pole_00.jpg                min3=  8.88 giữa=  8.88 cột=giữa  ok
  HAZ street_00.jpg              min3= 10.38 giữa= 10.38 cột=giữa  ok
  CLR street_01.jpg              min3= 17.17 giữa= 23.00 cột=trái  BÁO THỪA
  HAZ vehicle_00.jpg             min3=  7.80 giữa= 10.23 cột=phải  ok
  HAZ vehicle_01.jpg             min3= 14.45 giữa= 19.08 cột=trái  ok

--- FRAME SYNTHETIC (ngưỡng min-3-cột 20.00) ---
  HAZ synth_person_front.jpg     min3=  7.10 giữa=  7.10 cột=giữa  ok
  HAZ synth_pole_right.jpg       min3=  5.25 giữa=  5.33 cột=phải  ok
  CLR synthclear_dark.jpg        min3=  4.25 giữa=  4.34 cột=phải  BÁO THỪA
  CLR synthclear_grey.jpg        min3=  8.85 giữa=  8.85 cột=giữa  BÁO THỪA
  CLR synthclear_indoor.jpg      min3=  6.63 giữa=  6.63 cột=giữa  BÁO THỪA
  CLR synthclear_tiled.jpg       min3= 27.56 giữa= 27.74 cột=phải  ok
  CLR synthclear_warm.jpg        min3=  9.13 giữa=  9.13 cột=giữa  BÁO THỪA
  CLR synthfar_person.jpg        min3= 19.27 giữa= 19.27 cột=giữa  BÁO THỪA (âm khó)
  CLR synthfar_pole.jpg          min3= 25.99 giữa= 26.39 cột=phải  ok (âm khó)
  HAZ synthnear_pole.jpg         min3=  5.46 giữa=  5.46 cột=giữa  ok
  CLR synthside_pole.jpg         min3=  9.66 giữa=  9.67 cột=phải  BÁO THỪA (âm khó)
Kết quả thô -> logs/depth_outdoor.json
```

- Exit code: `0`

## Depth outdoor, p3 (bắt vật mảnh)

```text
$ .venv/bin/python scripts/11_depth_probe.py --pct 3 --dump logs/depth_outdoor_p3.json
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/287 [00:00<?, ?it/s]Loading weights: 100%|██████████| 287/287 [00:00<00:00, 3705.69it/s]
Depth : depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
Load  : 0.5s | VRAM 48 MiB | fp16
Frames: 27 (15 có vật cản / 12 trống) | 640px | p3 | mét

--- Latency depth (27 lần) ---
  median 11 ms | min/max 10/87 ms   (VLM 4B đang là ~845 ms)

--- Quét ngưỡng, min 3 cột (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%
       2.00m     0%   100%   100%    50%    0%
       3.00m     0%   100%   100%    50%    0%
       4.00m     0%   100%   100%    50%    0%
       5.00m     0%    92%   100%    46%    0%
       6.00m    13%    92%   100%    52%  100%
       8.00m    40%    83%   100%    62%   83%
      10.00m    60%    58%    75%    59%   78%
      12.00m    73%    58%    75%    66%   82%
      14.00m    80%    58%    75%    69%   75%
      15.00m    87%    58%    75%    73%   77%
      16.00m    87%    58%    75%    73%   77%
      18.00m    87%    50%    75%    68%   77%
      19.00m    93%    42%    50%    68%   71%
      19.50m    93%    42%    50%    68%   71%
      20.00m   100%    42%    50%    71%   73%  <- chọn
      21.00m   100%    42%    50%    71%   73%
      22.00m   100%    42%    50%    71%   73%
      24.00m   100%    33%    25%    67%   73%
      26.00m   100%    17%     0%    58%   73%
  quy tắc chọn: recall 100% + specificity cao nhất
  dư địa tới vật cản xa nhất (19.59): +0.41 = 2.0%   <- QUÁ SÁT, ngưỡng bị ghim bởi 1 frame

--- Quét ngưỡng, chỉ cột giữa (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%
       2.00m     0%   100%   100%    50%    0%
       3.00m     0%   100%   100%    50%    0%
       4.00m     0%   100%   100%    50%    0%
       5.00m     0%    92%   100%    46%    0%
       6.00m    13%    92%   100%    52%  100%
       8.00m    20%    83%   100%    52%  100%
      10.00m    33%    58%    75%    46%   80%
      12.00m    67%    58%    75%    62%   80%
      14.00m    67%    58%    75%    62%   80%
      15.00m    67%    58%    75%    62%   80%
      16.00m    67%    58%    75%    62%   80%
      18.00m    73%    58%    75%    66%   82%
      19.00m    87%    50%    50%    68%   77%
      19.50m    93%    50%    50%    72%   71%
      20.00m    93%    50%    50%    72%   71%
      21.00m   100%    50%    50%    75%   73%  <- chọn
      22.00m   100%    50%    50%    75%   73%
      24.00m   100%    42%    50%    71%   73%
      26.00m   100%    25%    25%    62%   73%
  quy tắc chọn: recall 100% + specificity cao nhất
  dư địa tới vật cản xa nhất (20.98): +0.02 = 0.1%   <- QUÁ SÁT, ngưỡng bị ghim bởi 1 frame

--- Thang đo này có thật là mét không? ---
  frame có vật cản :   5.23 ..  19.59
  frame lối trống  :   4.24 ..  52.81
  Vật cản gần nhất trong bộ frame cách chừng 1-2 bước, nhưng đọc ra 5.2 'mét'.
  Checkpoint metric train trên VKITTI (camera gắn trên xe, tầm 0-80m), nên camera
  cầm tay ở tầm người là out-of-distribution: sai số thang chừng 4-5 lần. Chỉ THỨ TỰ
  dùng được. §16 KHÔNG ngưỡng được theo '3 bước chân' — ngưỡng phải hiệu chuẩn
  cho từng độ cao/góc chúc camera và sẽ trôi khi đổi cấu hình.

--- So với baseline ---
  cấu hình                  recall  specif  sp.khó  balanc
  VLM 4B v6 (near+mid)       100%    58%     0%    79%
  VLM 4B v6 (chỉ near)        87%    75%    25%    81%
  depth min-3-cột @20.0      100%    42%    50%    71%
  depth cột-giữa @21.0       100%    50%    50%    75%

--- Cùng-1-cây-cột (phép thử VLM đã trượt) ---
  synthnear_pole.jpg     min3=  5.44 giữa=  5.44 3cột=[5.95, 5.44, 6.03] gần-nhất=giữa
  synthfar_pole.jpg      min3= 24.99 giữa= 25.76 3cột=[25.84, 25.76, 24.99] gần-nhất=phải
  synthside_pole.jpg     min3=  9.62 giữa=  9.65 3cột=[9.79, 9.65, 9.62] gần-nhất=phải
  cột-xa xa hơn cột-gần?          CÓ  (24.99 vs 5.44)
  cột-lệch-bên bị cột-giữa bỏ qua? CÓ  (9.65 vs 5.44)
  => Depth phân biệt được xa/gần, thứ VLM không làm được.

--- FRAME THẬT (ngưỡng min-3-cột 20.00) ---
  CLR clear_00.jpg               min3= 23.10 giữa= 33.48 cột=trái  ok (âm khó)
  CLR clear_01.jpg               min3= 31.44 giữa= 64.62 cột=phải  ok
  HAZ clear_02.jpg               min3= 18.17 giữa= 19.00 cột=phải  ok
  CLR clearpath_00.jpg           min3= 52.81 giữa= 52.81 cột=giữa  ok
  HAZ clearpath_01.jpg           min3= 10.57 giữa= 11.40 cột=phải  ok
  HAZ crowd_00.jpg               min3=  9.98 giữa= 11.53 cột=trái  ok
  HAZ crowd_01.jpg               min3= 13.40 giữa= 18.91 cột=trái  ok
  HAZ door_00.jpg                min3= 19.59 giữa= 20.98 cột=trái  ok
  HAZ door_01.jpg                min3=  7.80 giữa= 10.33 cột=phải  ok
  HAZ lowlight_00.jpg            min3=  6.84 giữa= 16.77 cột=trái  ok
  HAZ lowlight_01.jpg            min3=  8.12 giữa=  8.70 cột=phải  ok
  HAZ pole_00.jpg                min3=  8.81 giữa=  8.81 cột=giữa  ok
  HAZ street_00.jpg              min3= 10.30 giữa= 10.30 cột=giữa  ok
  CLR street_01.jpg              min3= 16.72 giữa= 22.42 cột=trái  BÁO THỪA
  HAZ vehicle_00.jpg             min3=  7.64 giữa= 10.16 cột=phải  ok
  HAZ vehicle_01.jpg             min3= 14.26 giữa= 18.57 cột=trái  ok

--- FRAME SYNTHETIC (ngưỡng min-3-cột 20.00) ---
  HAZ synth_person_front.jpg     min3=  7.07 giữa=  7.07 cột=giữa  ok
  HAZ synth_pole_right.jpg       min3=  5.23 giữa=  5.29 cột=phải  ok
  CLR synthclear_dark.jpg        min3=  4.24 giữa=  4.32 cột=phải  BÁO THỪA
  CLR synthclear_grey.jpg        min3=  8.83 giữa=  8.83 cột=giữa  BÁO THỪA
  CLR synthclear_indoor.jpg      min3=  6.59 giữa=  6.59 cột=giữa  BÁO THỪA
  CLR synthclear_tiled.jpg       min3= 25.04 giữa= 25.04 cột=giữa  ok
  CLR synthclear_warm.jpg        min3=  9.09 giữa=  9.09 cột=giữa  BÁO THỪA
  CLR synthfar_person.jpg        min3= 18.33 giữa= 18.33 cột=giữa  BÁO THỪA (âm khó)
  CLR synthfar_pole.jpg          min3= 24.99 giữa= 25.76 cột=phải  ok (âm khó)
  HAZ synthnear_pole.jpg         min3=  5.44 giữa=  5.44 cột=giữa  ok
  CLR synthside_pole.jpg         min3=  9.62 giữa=  9.65 cột=phải  BÁO THỪA (âm khó)
Kết quả thô -> logs/depth_outdoor_p3.json
```

- Exit code: `0`

## Tải checkpoint depth indoor

```text
$ env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python scripts/01_download_models.py --only depth_indoor

=== depth_indoor: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf ===
Fetching 3 files:   0%|          | 0/3 [00:00<?, ?it/s]Fetching 3 files: 100%|██████████| 3/3 [00:01<00:00,  2.07it/s]Fetching 3 files: 100%|██████████| 3/3 [00:01<00:00,  2.07it/s]
  -> /home/ubuntu/ai-assistant/models/hf/hub/models--depth-anything--Depth-Anything-V2-Metric-Indoor-Small-hf/snapshots/8078d68a9c75a972131914f6afd0c1723be0da7f  (2s)

HF_HOME = /home/ubuntu/ai-assistant/models/hf
Đã tải xong 1 model.
```

- Exit code: `0`

## Depth indoor (thang có sát tầm người hơn?)

```text
$ .venv/bin/python scripts/11_depth_probe.py --which indoor --dump logs/depth_indoor.json
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights:   0%|          | 0/287 [00:00<?, ?it/s]Loading weights: 100%|██████████| 287/287 [00:00<00:00, 3575.99it/s]
Depth : depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
Load  : 0.6s | VRAM 48 MiB | fp16
Frames: 27 (15 có vật cản / 12 trống) | 640px | p10 | mét

--- Latency depth (27 lần) ---
  median 10 ms | min/max 10/87 ms   (VLM 4B đang là ~845 ms)

--- Quét ngưỡng, min 3 cột (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%
       2.00m     7%   100%   100%    53%  100%
       3.00m    13%   100%   100%    57%  100%
       4.00m    47%    83%   100%    65%  100%
       5.00m    67%    83%   100%    75%   80%
       6.00m    67%    58%    75%    62%   80%
       8.00m    93%    42%    25%    68%   64%
      10.00m    93%    25%     0%    59%   64%
      12.00m   100%    25%     0%    62%   67%  <- chọn
      14.00m   100%    25%     0%    62%   67%
      15.00m   100%     8%     0%    54%   67%
      16.00m   100%     8%     0%    54%   67%
      18.00m   100%     8%     0%    54%   67%
      19.00m   100%     8%     0%    54%   67%
      19.50m   100%     8%     0%    54%   67%
      20.00m   100%     0%     0%    50%   67%
      21.00m   100%     0%     0%    50%   67%
      22.00m   100%     0%     0%    50%   67%
      24.00m   100%     0%     0%    50%   67%
      26.00m   100%     0%     0%    50%   67%
  quy tắc chọn: recall 100% + specificity cao nhất
  dư địa tới vật cản xa nhất (11.44): +0.56 = 4.7%   <- QUÁ SÁT, ngưỡng bị ghim bởi 1 frame

--- Quét ngưỡng, chỉ cột giữa (free < ngưỡng => cảnh báo) ---
      ngưỡng  recall  specif  sp.khó  balanc  hướng
       1.50m     0%   100%   100%    50%    0%
       2.00m     7%   100%   100%    53%  100%
       3.00m     7%   100%   100%    53%  100%
       4.00m    27%    83%   100%    55%  100%
       5.00m    47%    83%   100%    65%   86%
       6.00m    53%    58%    75%    56%   75%
       8.00m    73%    42%    25%    57%   73%
      10.00m    87%    33%    25%    60%   69%
      12.00m    93%    33%    25%    63%   64%
      14.00m   100%    25%     0%    62%   67%  <- chọn
      15.00m   100%    25%     0%    62%   67%
      16.00m   100%    17%     0%    58%   67%
      18.00m   100%    17%     0%    58%   67%
      19.00m   100%    17%     0%    58%   67%
      19.50m   100%    17%     0%    58%   67%
      20.00m   100%     0%     0%    50%   67%
      21.00m   100%     0%     0%    50%   67%
      22.00m   100%     0%     0%    50%   67%
      24.00m   100%     0%     0%    50%   67%
      26.00m   100%     0%     0%    50%   67%
  quy tắc chọn: recall 100% + specificity cao nhất
  dư địa tới vật cản xa nhất (12.12): +1.88 = 13.4%

--- Thang đo này có thật là mét không? ---
  frame có vật cản :   1.68 ..  11.44
  frame lối trống  :   3.19 ..  19.73
  Vật cản gần nhất trong bộ frame cách chừng 1-2 bước, nhưng đọc ra 1.7 'mét'.
  Checkpoint metric train trên VKITTI (camera gắn trên xe, tầm 0-80m), nên camera
  cầm tay ở tầm người là out-of-distribution: sai số thang chừng 4-5 lần. Chỉ THỨ TỰ
  dùng được. §16 KHÔNG ngưỡng được theo '3 bước chân' — ngưỡng phải hiệu chuẩn
  cho từng độ cao/góc chúc camera và sẽ trôi khi đổi cấu hình.

--- So với baseline ---
  cấu hình                  recall  specif  sp.khó  balanc
  VLM 4B v6 (near+mid)       100%    58%     0%    79%
  VLM 4B v6 (chỉ near)        87%    75%    25%    81%
  depth min-3-cột @12.0      100%    25%     0%    62%
  depth cột-giữa @14.0       100%    25%     0%    62%

--- Cùng-1-cây-cột (phép thử VLM đã trượt) ---
  synthnear_pole.jpg     min3=  1.68 giữa=  1.68 3cột=[3.73, 1.68, 3.76] gần-nhất=giữa
  synthfar_pole.jpg      min3=  5.79 giữa=  5.79 3cột=[7.21, 5.79, 7.09] gần-nhất=giữa
  synthside_pole.jpg     min3=  7.50 giữa=  7.55 3cột=[7.57, 7.55, 7.5] gần-nhất=phải
  cột-xa xa hơn cột-gần?          CÓ  (5.79 vs 1.68)
  cột-lệch-bên bị cột-giữa bỏ qua? CÓ  (7.55 vs 1.68)
  => Depth phân biệt được xa/gần, thứ VLM không làm được.

--- FRAME THẬT (ngưỡng min-3-cột 12.00) ---
  CLR clear_00.jpg               min3=  8.54 giữa= 12.36 cột=trái  BÁO THỪA (âm khó)
  CLR clear_01.jpg               min3= 14.20 giữa= 19.55 cột=phải  ok
  HAZ clear_02.jpg               min3=  6.28 giữa=  8.24 cột=phải  ok
  CLR clearpath_00.jpg           min3= 19.73 giữa= 19.81 cột=trái  ok
  HAZ clearpath_01.jpg           min3=  2.29 giữa=  6.64 cột=phải  ok
  HAZ crowd_00.jpg               min3=  3.77 giữa=  3.99 cột=phải  ok
  HAZ crowd_01.jpg               min3=  6.38 giữa= 10.78 cột=trái  ok
  HAZ door_00.jpg                min3= 11.44 giữa= 12.12 cột=trái  ok
  HAZ door_01.jpg                min3=  4.50 giữa=  4.60 cột=phải  ok
  HAZ lowlight_00.jpg            min3=  3.88 giữa=  9.68 cột=trái  ok
  HAZ lowlight_01.jpg            min3=  6.06 giữa=  6.31 cột=phải  ok
  HAZ pole_00.jpg                min3=  3.86 giữa=  3.86 cột=giữa  ok
  HAZ street_00.jpg              min3=  3.85 giữa=  3.85 cột=giữa  ok
  CLR street_01.jpg              min3= 14.80 giữa= 15.45 cột=trái  ok
  HAZ vehicle_00.jpg             min3=  3.38 giữa=  4.09 cột=phải  ok
  HAZ vehicle_01.jpg             min3=  6.12 giữa=  6.12 cột=giữa  ok

--- FRAME SYNTHETIC (ngưỡng min-3-cột 12.00) ---
  HAZ synth_person_front.jpg     min3=  4.79 giữa=  4.79 cột=giữa  ok
  HAZ synth_pole_right.jpg       min3=  4.96 giữa=  5.18 cột=trái  ok
  CLR synthclear_dark.jpg        min3=  5.54 giữa=  5.56 cột=phải  BÁO THỪA
  CLR synthclear_grey.jpg        min3=  5.74 giữa=  5.74 cột=giữa  BÁO THỪA
  CLR synthclear_indoor.jpg      min3=  3.19 giữa=  3.21 cột=phải  BÁO THỪA
  CLR synthclear_tiled.jpg       min3=  9.14 giữa=  9.14 cột=giữa  BÁO THỪA
  CLR synthclear_warm.jpg        min3=  3.78 giữa=  3.80 cột=trái  BÁO THỪA
  CLR synthfar_person.jpg        min3=  6.08 giữa=  6.08 cột=giữa  BÁO THỪA (âm khó)
  CLR synthfar_pole.jpg          min3=  5.79 giữa=  5.79 cột=giữa  BÁO THỪA (âm khó)
  HAZ synthnear_pole.jpg         min3=  1.68 giữa=  1.68 cột=giữa  ok
  CLR synthside_pole.jpg         min3=  7.50 giữa=  7.55 cột=phải  BÁO THỪA (âm khó)
Kết quả thô -> logs/depth_indoor.json
```

- Exit code: `0`

## Ghép VLM + depth indoor

```text
$ .venv/bin/python scripts/12_fuse_vlm_depth.py logs/eval_vlm_qwen3vl4b.json logs/depth_indoor.json
VLM   : Qwen/Qwen3-VL-4B-Instruct / pipe_v6
Depth : depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf | free3 | p10 | 10 ms
Frames: 27 khớp cả hai dump

--- Quét ngưỡng ghép (free3) ---
    ngưỡng  D recall  D spec  D khó  E recall  E spec  E khó
     8.00      93%     75%    25%     100%     75%    25%
    10.00      93%     67%     0%     100%     67%     0%
    12.00     100%     67%     0%     100%     67%     0%
    14.00     100%     67%     0%     100%     67%     0%
    15.00     100%     58%     0%     100%     58%     0%
    16.00     100%     58%     0%     100%     58%     0%
    18.00     100%     58%     0%     100%     58%     0%
    19.00     100%     58%     0%     100%     58%     0%
    19.50     100%     58%     0%     100%     58%     0%
    20.00     100%     58%     0%     100%     58%     0%
    21.00     100%     58%     0%     100%     58%     0%
    22.00     100%     58%     0%     100%     58%     0%
    24.00     100%     58%     0%     100%     58%     0%
    26.00     100%     58%     0%     100%     58%     0%
  quy tắc chọn: E giữ recall 100% ở ngưỡng thấp nhất
  => ngưỡng 8.00m (recall 100%, specificity 75%, âm khó 25%)
  dư địa tới vật cản xa nhất (11.44): -3.44 = -43.0%   <- QUÁ SÁT, ngưỡng đang bị ghim bởi 1 frame

Ngưỡng dùng: free3 < 8.00m

chiến lược          recall  specif  sp.khó   balanc   type  hướng
A. VLM near+mid      100%    58%     0%     79%   80%   80%
B. VLM chỉ near       87%    75%    25%     81%   85%   85%
C. chỉ depth          93%    42%    25%     68%    0%   64%
D. VLM AND depth      93%    75%    25%     84%   79%   79%
E. D + cửa thoát     100%    75%    25%     88%   80%   80%

A. VLM near+mid
   bỏ sót  (0): —
   báo thừa(5): clear_00.jpg (âm khó), clear_01.jpg, synthfar_pole.jpg (âm khó), synthfar_person.jpg (âm khó), synthside_pole.jpg (âm khó)
B. VLM chỉ near
   bỏ sót  (2): clear_02.jpg, vehicle_00.jpg
   báo thừa(3): clear_00.jpg (âm khó), synthfar_pole.jpg (âm khó), synthside_pole.jpg (âm khó)
C. chỉ depth
   bỏ sót  (1): door_00.jpg
   báo thừa(7): synthclear_grey.jpg, synthclear_warm.jpg, synthclear_dark.jpg, synthclear_indoor.jpg, synthfar_pole.jpg (âm khó), synthfar_person.jpg (âm khó), synthside_pole.jpg (âm khó)
D. VLM AND depth
   bỏ sót  (1): door_00.jpg
   báo thừa(3): synthfar_pole.jpg (âm khó), synthfar_person.jpg (âm khó), synthside_pole.jpg (âm khó)
E. D + cửa thoát
   bỏ sót  (0): —
   báo thừa(3): synthfar_pole.jpg (âm khó), synthfar_person.jpg (âm khó), synthside_pole.jpg (âm khó)

=== Kết luận ===
                  recall  specif  sp.khó  balanc
  A baseline       100%    58%     0%    79%
  D AND thuần       93%    75%    25%    84%
  E đề xuất        100%    75%    25%    88%
  âm khó: 0% -> 25%   <- đây là chỗ VLM đơn độc bằng 0
  => Recall giữ nguyên 100%, specificity 58% -> 75%, âm khó 0% -> 25%.
     Không đổi gì lấy gì: depth cấp đúng tín hiệu VLM đang thiếu.
  Cửa thoát cho door/hole/step cứu được 1 frame mà AND thuần bỏ sót, giá: không mất specificity.

LƯU Ý: 12 frame âm — specificity vẫn nhảy từng bước lớn. §8 vẫn cần 100-500 frame thật.
LƯU Ý: ngưỡng trên KHÔNG phải mét thật (checkpoint VKITTI, camera gắn xe) — chỉ
       dùng được thứ tự, và phải hiệu chuẩn lại cho từng máy.
```

- Exit code: `0`

## 08_ab_prompt.py (chỉ đo tốc độ, xác nhận bản sửa)

```text
$ .venv/bin/python scripts/08_ab_prompt.py
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in /home/ubuntu/ai-assistant/.venv/lib/python3.14/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py.
[ERROR] `max_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in /home/ubuntu/ai-assistant/.venv/lib/python3.14/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py.
Loading weights:   0%|          | 0/713 [00:00<?, ?it/s]Loading weights:   0%|          | 1/713 [00:00<10:24,  1.14it/s]Loading weights:   8%|▊         | 58/713 [00:00<00:08, 80.59it/s]Loading weights:  13%|█▎        | 91/713 [00:01<00:06, 95.35it/s]Loading weights:  16%|█▌        | 115/713 [00:01<00:06, 92.49it/s]Loading weights:  19%|█▉        | 135/713 [00:01<00:05, 98.64it/s]Loading weights:  21%|██        | 151/713 [00:01<00:05, 105.00it/s]Loading weights:  24%|██▎       | 168/713 [00:01<00:05, 102.16it/s]Loading weights:  26%|██▌       | 182/713 [00:02<00:05, 104.38it/s]Loading weights:  28%|██▊       | 201/713 [00:02<00:04, 111.33it/s]Loading weights:  30%|███       | 214/713 [00:02<00:04, 109.78it/s]Loading weights:  32%|███▏      | 226/713 [00:02<00:04, 103.14it/s]Loading weights:  34%|███▍      | 245/713 [00:02<00:04, 109.76it/s]Loading weights:  37%|███▋      | 264/713 [00:02<00:03, 125.55it/s]Loading weights:  39%|███▉      | 278/713 [00:02<00:03, 112.56it/s]Loading weights:  41%|████      | 290/713 [00:03<00:03, 111.63it/s]Loading weights:  42%|████▏     | 302/713 [00:03<00:03, 105.05it/s]Loading weights:  44%|████▍     | 313/713 [00:03<00:04, 98.30it/s] Loading weights:  47%|████▋     | 333/713 [00:03<00:03, 115.28it/s]Loading weights:  48%|████▊     | 345/713 [00:03<00:03, 108.79it/s]Loading weights:  50%|█████     | 357/713 [00:03<00:03, 104.15it/s]Loading weights:  53%|█████▎    | 377/713 [00:03<00:03, 111.44it/s]Loading weights:  55%|█████▍    | 389/713 [00:04<00:03, 105.23it/s]Loading weights:  73%|███████▎  | 522/713 [00:04<00:00, 396.14it/s]Loading weights:  89%|████████▊ | 632/713 [00:04<00:00, 571.92it/s]Loading weights:  98%|█████████▊| 700/713 [00:04<00:00, 586.44it/s]Loading weights: 100%|██████████| 713/713 [00:04<00:00, 164.75it/s]
Qwen/Qwen3-VL-4B-Instruct | fp16 | 31 frame | repeat=1

--- pipe_v1 ---
  latency median 650 ms | token mean 7.8 | CLEAR 12/31 (39%) | sai schema 0
    clear_00.jpg               CLEAR
    clear_01.jpg               vehicle|front_right|stop
    clear_02.jpg               person|left|stop
    clearpath_00.jpg           CLEAR
    clearpath_01.jpg           pole|front_right|move_left
    clearpath_02.jpg           hole|front|stop
    crowd_00.jpg               vehicle|front_left|stop
    crowd_01.jpg               vehicle|front|stop
    door_00.jpg                CLEAR
    door_01.jpg                door|front|stop
    lowlight_00.jpg            vehicle|front_right|stop
    lowlight_01.jpg            CLEAR
    pole_00.jpg                pole|front_right|move_left
    stairs_00.jpg              pole|front_left|move_left
    stairs_01.jpg              CLEAR
    street_00.jpg              vehicle|front|stop
    street_01.jpg              vehicle|front|stop
    synth_clear.jpg            CLEAR
    synth_person_front.jpg     pole|front|stop
    synth_pole_right.jpg       pole|front_right|move_left
    synthclear_dark.jpg        CLEAR
    synthclear_grey.jpg        CLEAR
    synthclear_indoor.jpg      CLEAR
    synthclear_tiled.jpg       CLEAR
    synthclear_warm.jpg        CLEAR
    synthfar_person.jpg        CLEAR
    synthfar_pole.jpg          pole|front|stop
    synthnear_pole.jpg         pole|front|stop
    synthside_pole.jpg         pole|front_left|move_left
    vehicle_00.jpg             object|left|move_left
    vehicle_01.jpg             vehicle|front_left|stop

--- pipe_v2 ---
  latency median 572 ms | token mean 6.5 | CLEAR 26/31 (84%) | sai schema 0
    clear_00.jpg               CLEAR
    clear_01.jpg               CLEAR
    clear_02.jpg               person|left|slow
    clearpath_00.jpg           CLEAR
    clearpath_01.jpg           CLEAR
    clearpath_02.jpg           hole|front|stop
    crowd_00.jpg               CLEAR
    crowd_01.jpg               CLEAR
    door_00.jpg                CLEAR
    door_01.jpg                CLEAR
    lowlight_00.jpg            CLEAR
    lowlight_01.jpg            CLEAR
    pole_00.jpg                CLEAR
    stairs_00.jpg              CLEAR
    stairs_01.jpg              CLEAR
    street_00.jpg              CLEAR
    street_01.jpg              CLEAR
    synth_clear.jpg            CLEAR
    synth_person_front.jpg     CLEAR
    synth_pole_right.jpg       pole|front_right|move_left
    synthclear_dark.jpg        CLEAR
    synthclear_grey.jpg        CLEAR
    synthclear_indoor.jpg      CLEAR
    synthclear_tiled.jpg       CLEAR
    synthclear_warm.jpg        CLEAR
    synthfar_person.jpg        CLEAR
    synthfar_pole.jpg          CLEAR
    synthnear_pole.jpg         pole|front|move_left
    synthside_pole.jpg         pole|front_left|move_left
    vehicle_00.jpg             CLEAR
    vehicle_01.jpg             CLEAR

--- pipe_v3 ---
  latency median 605 ms | token mean 7.1 | CLEAR 19/31 (61%) | sai schema 0
    clear_00.jpg               CLEAR
    clear_01.jpg               CLEAR
    clear_02.jpg               person|front|slow
    clearpath_00.jpg           CLEAR
    clearpath_01.jpg           step|front|slow
    clearpath_02.jpg           hole|front|stop
    crowd_00.jpg               CLEAR
    crowd_01.jpg               CLEAR
    door_00.jpg                CLEAR
    door_01.jpg                door|front|move_left
    lowlight_00.jpg            person|front_right|slow
    lowlight_01.jpg            CLEAR
    pole_00.jpg                pole|front|move_left
    stairs_00.jpg              CLEAR
    stairs_01.jpg              CLEAR
    street_00.jpg              vehicle|front|stop
    street_01.jpg              CLEAR
    synth_clear.jpg            CLEAR
    synth_person_front.jpg     pole|front|move_left
    synth_pole_right.jpg       pole|front_right|move_left
    synthclear_dark.jpg        CLEAR
    synthclear_grey.jpg        CLEAR
    synthclear_indoor.jpg      CLEAR
    synthclear_tiled.jpg       CLEAR
    synthclear_warm.jpg        CLEAR
    synthfar_person.jpg        CLEAR
    synthfar_pole.jpg          pole|front|move_left
    synthnear_pole.jpg         pole|front|move_left
    synthside_pole.jpg         pole|front_left|move_left
    vehicle_00.jpg             CLEAR
    vehicle_01.jpg             CLEAR

--- pipe_v4 ---
  latency median 602 ms | token mean 6.9 | CLEAR 19/31 (61%) | sai schema 0
    clear_00.jpg               CLEAR
    clear_01.jpg               CLEAR
    clear_02.jpg               person|front|slow
    clearpath_00.jpg           CLEAR
    clearpath_01.jpg           object|front|stop
    clearpath_02.jpg           CLEAR
    crowd_00.jpg               object|front|stop
    crowd_01.jpg               CLEAR
    door_00.jpg                CLEAR
    door_01.jpg                CLEAR
    lowlight_00.jpg            CLEAR
    lowlight_01.jpg            CLEAR
    pole_00.jpg                vehicle|front|stop
    stairs_00.jpg              CLEAR
    stairs_01.jpg              CLEAR
    street_00.jpg              CLEAR
    street_01.jpg              vehicle|front|stop
    synth_clear.jpg            CLEAR
    synth_person_front.jpg     person|front|stop
    synth_pole_right.jpg       pole|front_right|move_left
    synthclear_dark.jpg        CLEAR
    synthclear_grey.jpg        CLEAR
    synthclear_indoor.jpg      CLEAR
    synthclear_tiled.jpg       CLEAR
    synthclear_warm.jpg        CLEAR
    synthfar_person.jpg        object|front|slow
    synthfar_pole.jpg          pole|front|stop
    synthnear_pole.jpg         pole|front|stop
    synthside_pole.jpg         pole|front_left|move_left
    vehicle_00.jpg             object|front|stop
    vehicle_01.jpg             CLEAR

--- pipe_v5 ---
  latency median 722 ms | token mean 9.6 | CLEAR 16/31 (52%) | sai schema 0
    clear_00.jpg               object|left|slow
    clear_01.jpg               CLEAR
    clear_02.jpg               person|front|stop
    clearpath_00.jpg           CLEAR
    clearpath_01.jpg           object|front|slow
    clearpath_02.jpg           object|front|slow
    crowd_00.jpg               person|left|slow
    crowd_01.jpg               CLEAR
    door_00.jpg                vehicle|right|stop
    door_01.jpg                CLEAR
    lowlight_00.jpg            CLEAR
    lowlight_01.jpg            CLEAR
    pole_00.jpg                pole|front_right|move_left
    stairs_00.jpg              object|left|stop
    stairs_01.jpg              CLEAR
    street_00.jpg              person|front|slow
    street_01.jpg              CLEAR
    synth_clear.jpg            CLEAR
    synth_person_front.jpg     person|front|stop
    synth_pole_right.jpg       pole|front_right|move_left
    synthclear_dark.jpg        CLEAR
    synthclear_grey.jpg        CLEAR
    synthclear_indoor.jpg      CLEAR
    synthclear_tiled.jpg       CLEAR
    synthclear_warm.jpg        CLEAR
    synthfar_person.jpg        CLEAR
    synthfar_pole.jpg          CLEAR
    synthnear_pole.jpg         pole|front|move_left
    synthside_pole.jpg         pole|front_left|move_left
    vehicle_00.jpg             object|left|slow
    vehicle_01.jpg             object|front|slow

--- pipe_v6 ---
  latency median 847 ms | token mean 10.3 | CLEAR 8/31 (26%) | sai schema 0
    clear_00.jpg               object|front_left|move_left
    clear_01.jpg               vehicle|front_right|slow
    clear_02.jpg               person|front|slow
    clearpath_00.jpg           CLEAR
    clearpath_01.jpg           object|front|stop
    clearpath_02.jpg           object|front|move_left
    crowd_00.jpg               person|left|none
    crowd_01.jpg               person|front|slow
    door_00.jpg                door|front|slow
    door_01.jpg                door|front|move_left
    lowlight_00.jpg            person|front_left|move_left
    lowlight_01.jpg            person|front|slow
    pole_00.jpg                pole|left|none
    stairs_00.jpg              pole|left|none
    stairs_01.jpg              person|front_left|move_left
    street_00.jpg              person|front_left|slow
    street_01.jpg              CLEAR
    synth_clear.jpg            CLEAR
    synth_person_front.jpg     person|front|slow
    synth_pole_right.jpg       pole|right|move_left
    synthclear_dark.jpg        CLEAR
    synthclear_grey.jpg        CLEAR
    synthclear_indoor.jpg      CLEAR
    synthclear_tiled.jpg       CLEAR
    synthclear_warm.jpg        CLEAR
    synthfar_person.jpg        object|front|none
    synthfar_pole.jpg          pole|front|move_left
    synthnear_pole.jpg         pole|front|move_left
    synthside_pole.jpg         pole|left|move_right
    vehicle_00.jpg             object|left|move_left
    vehicle_01.jpg             object|front_left|move_left

=== Tổng hợp (chỉ latency — KHÔNG phải bảng chọn prompt) ===
prompt       latency  token  CLEAR rate  sai schema
pipe_v1         650ms    7.8        39%           0
pipe_v2         572ms    6.5        84%           0
pipe_v3         605ms    7.1        61%           0
pipe_v4         602ms    6.9        61%           0
pipe_v5         722ms    9.6        52%           0
pipe_v6         847ms   10.3        26%           0

Nhanh nhất: pipe_v2 — 572 ms, 6.5 token.
Nhanh KHÔNG có nghĩa là đúng. Chốt prompt bằng 09_eval_vlm.py.
```

- Exit code: `0`

- Finished: `2026-08-24 07:28:41 UTC`
