# Benchmark report

- Mode: `project`
- Started: `2026-08-24 08:11:59 UTC`

## Dataset checks

| Dataset | Status | Details |
|---|---|---|
| `diode` | `ok` | size_bytes=9173067334, depth_arrays=1542, rgb_images=771 |
| `nyu_depth_v2` | `ok` | size_bytes=2972037809, keys=['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes'], missing_keys=[], images_shape=[1449, 3, 640, 480], depths_shape=[1449, 640, 480] |
| `sun_rgbd` | `ok` | size_bytes=7174366246, files=285925 |
| `coco` | `ok` | size_bytes=1087323831, annotation_files=['annotations/captions_train2017.json', 'annotations/captions_val2017.json', 'annotations/instances_train2017.json', 'annotations/instances_val2017.json', 'annotations/person_keypoints_train2017.json', 'annotations/person_keypoints_val2017.json'] |
| `fleurs_vi` | `ok` | size_bytes=3061201006, cached_files=9, validation_rows=361 |
| `ai2thor` | `ok` | package=ai2thor |

## Project benchmark commands

### vlm_eval_4b

- Exit code: `0`
- Duration: `185.524 s`

```text
[ERROR] `min_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in /home/ubuntu/ai-assistant/.venv/lib/python3.14/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py.
[ERROR] `max_frames` is part of Qwen3VLVideoProcessorInitKwargs, but not documented. Make sure to add it to the docstring of the function in /home/ubuntu/ai-assistant/.venv/lib/python3.14/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py.
Model : Qwen/Qwen3-VL-4B-Instruct | fp16 | 640px
Frames: 27 dùng được (15 có vật cản / 12 trống)
Prompt: pipe_v1, pipe_v2, pipe_v3, pipe_v4, pipe_v5, pipe_v6 | repeat=1

--- pipe_v1 ---
  recall 87% (13/15)  specificity 67% (8/12)  balanced 77%
  specificity trên 4 frame ÂM KHÓ (vật ở xa / lệch bên, §16): 50% (2/4)
  type 77%  position 77%  | 646 ms  7.9 tok  sai schema 0
  cùng-1-cây-cột: gần=pole|front|stop xa=pole|front|stop lệch-bên=pole|front_left|move_left
                  -> KHÔNG phân biệt được khoảng cách/vị trí
    CLR clear_00.jpg               clear|none|none                    ok
    CLR clear_01.jpg               vehicle|front_right|stop           BÁO THỪA
    HAZ clear_02.jpg               person|left|stop                   ok
    CLR clearpath_00.jpg           clear|none|none                    ok
    HAZ clearpath_01.jpg           pole|front_right|move_left         ok
    HAZ crowd_00.jpg               vehicle|front_left|stop            ok
    HAZ crowd_01.jpg               vehicle|front|stop                 ok
    HAZ door_00.jpg                clear|none|none                    BỎ SÓT
    HAZ door_01.jpg                door|front|stop                    ok
    HAZ lowlight_00.jpg            vehicle|front_right|stop           ok
    HAZ lowlight_01.jpg            clear|none|none                    BỎ SÓT
    HAZ pole_00.jpg                pole|front_right|move_left         ok
    HAZ street_00.jpg              vehicle|front|stop                 ok
    CLR street_01.jpg              vehicle|front|stop                 BÁO THỪA
    HAZ synth_person_front.jpg     pole|front|stop                    ok
    HAZ synth_pole_right.jpg       pole|front_right|move_left         ok
    CLR synthclear_dark.jpg        clear|none|none                    ok
    CLR synthclear_grey.jpg        clear|none|none                    ok
    CLR synthclear_indoor.jpg      clear|none|none                    ok
    CLR synthclear_tiled.jpg       clear|none|none                    ok
    CLR synthclear_warm.jpg        clear|none|none                    ok
    CLR synthfar_person.jpg        clear|none|none                    ok
    CLR synthfar_pole.jpg          pole|front|stop                    BÁO THỪA
    HAZ synthnear_pole.jpg         pole|front|stop                    ok
    CLR synthside_pole.jpg         pole|front_left|move_left          BÁO THỪA
    HAZ vehicle_00.jpg             object|left|move_left              ok
    HAZ vehicle_01.jpg             vehicle|front_left|stop            ok

--- pipe_v2 ---
  recall 20% (3/15)  specificity 92% (11/12)  balanced 56%
  specificity trên 4 frame ÂM KHÓ (vật ở xa / lệch bên, §16): 75% (3/4)
  type 100%  position 67%  | 568 ms  6.4 tok  sai schema 0
  cùng-1-cây-cột: gần=pole|front|move_left xa=clear|none|none lệch-bên=pole|front_left|move_left
                  -> phân biệt được
    CLR clear_00.jpg               clear|none|none                    ok
    CLR clear_01.jpg               clear|none|none                    ok
    HAZ clear_02.jpg               person|left|slow                   ok
    CLR clearpath_00.jpg           clear|none|none                    ok
    HAZ clearpath_01.jpg           clear|none|none                    BỎ SÓT
    HAZ crowd_00.jpg               clear|none|none                    BỎ SÓT
    HAZ crowd_01.jpg               clear|none|none                    BỎ SÓT
    HAZ door_00.jpg                clear|none|none                    BỎ SÓT
    HAZ door_01.jpg                clear|none|none                    BỎ SÓT
    HAZ lowlight_00.jpg            clear|none|none                    BỎ SÓT
    HAZ lowlight_01.jpg            clear|none|none                    BỎ SÓT
    HAZ pole_00.jpg                clear|none|none                    BỎ SÓT
    HAZ street_00.jpg              clear|none|none                    BỎ SÓT
    CLR street_01.jpg              clear|none|none                    ok
    HAZ synth_person_front.jpg     clear|none|none                    BỎ SÓT
    HAZ synth_pole_right.jpg       pole|front_right|move_left         ok
    CLR synthclear_dark.jpg        clear|none|none                    ok
    CLR synthclear_grey.jpg        clear|none|none                    ok
    CLR synthclear_indoor.jpg      clear|none|none                    ok
    CLR synthclear_tiled.jpg       clear|none|none                    ok
    CLR synthclear_warm.jpg        clear|none|none                    ok
    CLR synthfar_person.jpg        clear|none|none                    ok
    CLR synthfar_pole.jpg          clear|none|none                    ok
    HAZ synthnear_pole.jpg         pole|front|move_left               ok
    CLR synthside_pole.jpg         pole|front_left|move_left          BÁO THỪA
    HAZ vehicle_00.jpg             clear|none|none                    BỎ SÓT
    HAZ vehicle_01.jpg             clear|none|none                    BỎ SÓT

--- pipe_v3 ---
  recall 60% (9/15)  specificity 83% (10/12)  balanced 72%
  specificity trên 4 frame ÂM KHÓ (vật ở xa / lệch bên, §16): 50% (2/4)
  type 78%  position 89%  | 605 ms  7.2 tok  sai schema 0
  cùng-1-cây-cột: gần=pole|front|move_left xa=pole|front|move_left lệch-bên=pole|front_left|move_left
                  -> KHÔNG phân biệt được khoảng cách/vị trí
    CLR clear_00.jpg               clear|none|none                    ok
    CLR clear_01.jpg               clear|none|none                    ok
    HAZ clear_02.jpg               person|front|slow                  ok
    CLR clearpath_00.jpg           clear|none|none                    ok
    HAZ clearpath_01.jpg           step|front|slow                    ok
    HAZ crowd_00.jpg               clear|none|none                    BỎ SÓT
    HAZ crowd_01.jpg               clear|none|none                    BỎ SÓT
    HAZ door_00.jpg                clear|none|none                    BỎ SÓT
    HAZ door_01.jpg                door|front|move_left               ok
    HAZ lowlight_00.jpg            person|front_right|slow            ok
    HAZ lowlight_01.jpg            clear|none|none                    BỎ SÓT
    HAZ pole_00.jpg                pole|front|move_left               ok
    HAZ street_00.jpg              vehicle|front|stop                 ok
    CLR street_01.jpg              clear|none|none                    ok
    HAZ synth_person_front.jpg     pole|front|move_left               ok
    HAZ synth_pole_right.jpg       pole|front_right|move_left         ok
    CLR synthclear_dark.jpg        clear|none|none                    ok
    CLR synthclear_grey.jpg        clear|none|none                    ok
    CLR synthclear_indoor.jpg      clear|none|none                    ok
    CLR synthclear_tiled.jpg       clear|none|none                    ok
    CLR synthclear_warm.jpg        clear|none|none                    ok
    CLR synthfar_person.jpg        clear|none|none                    ok
    CLR synthfar_pole.jpg          pole|front|move_left               BÁO THỪA
    HAZ synthnear_pole.jpg         pole|front|move_left               ok
    CLR synthside_pole.jpg         pole|front_left|move_left          BÁO THỪA
    HAZ vehicle_00.jpg             clear|none|none                    BỎ SÓT
    HAZ vehicle_01.jpg             clear|none|none                    BỎ SÓT

--- pipe_v4 ---
  recall 53% (8/15)  specificity 67% (8/12)  balanced 60%
  specificity trên 4 frame ÂM KHÓ (vật ở xa / lệch bên, §16): 25% (1/4)
  type 75%  position 88%  | 600 ms  7.0 tok  sai schema 0
  cùng-1-cây-cột: gần=pole|front|stop xa=pole|front|stop lệch-bên=pole|front_left|move_left
                  -> KHÔNG phân biệt được khoảng cách/vị trí
    CLR clear_00.jpg               clear|none|none                    ok
    CLR clear_01.jpg               clear|front|move_left              ok
    HAZ clear_02.jpg               person|front|slow                  ok
    CLR clearpath_00.jpg           clear|none|none                    ok
    HAZ clearpath_01.jpg           object|front|stop                  ok
    HAZ crowd_00.jpg               object|front|stop                  ok
    HAZ crowd_01.jpg               clear|none|none                    BỎ SÓT
    HAZ door_00.jpg                clear|none|none                    BỎ SÓT
    HAZ door_01.jpg                clear|none|none                    BỎ SÓT
    HAZ lowlight_00.jpg            clear|none|none                    BỎ SÓT
    HAZ lowlight_01.jpg            clear|none|none                    BỎ SÓT
    HAZ pole_00.jpg                vehicle|front|stop                 ok
    HAZ street_00.jpg              clear|none|none                    BỎ SÓT
    CLR street_01.jpg              vehicle|front|stop                 BÁO THỪA
    HAZ synth_person_front.jpg     person|front|stop                  ok
    HAZ synth_pole_right.jpg       pole|front_right|move_left         ok
    CLR synthclear_dark.jpg        clear|none|none                    ok
    CLR synthclear_grey.jpg        clear|none|none                    ok
    CLR synthclear_indoor.jpg      clear|none|none                    ok
    CLR synthclear_tiled.jpg       clear|none|none                    ok
    CLR synthclear_warm.jpg        clear|none|none                    ok
    CLR synthfar_person.jpg        object|front|slow                  BÁO THỪA
    CLR synthfar_pole.jpg          pole|front|stop                    BÁO THỪA
    HAZ synthnear_pole.jpg         pole|front|stop                    ok
    CLR synthside_pole.jpg         pole|front_left|move_left          BÁO THỪA
    HAZ vehicle_00.jpg             object|front|stop                  ok
    HAZ vehicle_01.jpg             clear|none|none                    BỎ SÓT

--- pipe_v5 ---
  recall 73% (11/15)  specificity 83% (10/12)  balanced 78%
  specificity trên 4 frame ÂM KHÓ (vật ở xa / lệch bên, §16): 50% (2/4)
  type 73%  position 73%  | 714 ms  9.6 tok  sai schema 0
  cùng-1-cây-cột: gần=pole|front|move_left xa=clear|none|none lệch-bên=pole|front_left|move_left
                  -> phân biệt được
    CLR clear_00.jpg               object|left|slow                   BÁO THỪA
    CLR clear_01.jpg               clear|none|none                    ok
    HAZ clear_02.jpg               person|front|stop                  ok
    CLR clearpath_00.jpg           clear|none|none                    ok
    HAZ clearpath_01.jpg           object|front|slow                  ok
    HAZ crowd_00.jpg               person|left|slow                   ok
    HAZ crowd_01.jpg               clear|none|none                    BỎ SÓT
    HAZ door_00.jpg                vehicle|right|stop                 ok
    HAZ door_01.jpg                clear|none|none                    BỎ SÓT
    HAZ lowlight_00.jpg            clear|none|none                    BỎ SÓT
    HAZ lowlight_01.jpg            clear|none|none                    BỎ SÓT
    HAZ pole_00.jpg                pole|front_right|move_left         ok
    HAZ street_00.jpg              person|front|slow                  ok
    CLR street_01.jpg              clear|none|none                    ok
    HAZ synth_person_front.jpg     person|front|stop                  ok
    HAZ synth_pole_right.jpg       pole|front_right|move_left         ok
    CLR synthclear_dark.jpg        clear|none|none                    ok
    CLR synthclear_grey.jpg        clear|none|none                    ok
    CLR synthclear_indoor.jpg      clear|none|none                    ok
    CLR synthclear_tiled.jpg       clear|none|none                    ok
    CLR synthclear_warm.jpg        clear|none|none                    ok
    CLR synthfar_person.jpg        clear|none|none                    ok
    CLR synthfar_pole.jpg          clear|none|none                    ok
    HAZ synthnear_pole.jpg         pole|front|move_left               ok
    CLR synthside_pole.jpg         pole|front_left|move_left          BÁO THỪA
    HAZ vehicle_00.jpg             object|left|slow                   ok
    HAZ vehicle_01.jpg             object|front|slow                  ok

--- pipe_v6 ---
  recall 87% (13/15)  specificity 75% (9/12)  balanced 81%
  specificity trên 4 frame ÂM KHÓ (vật ở xa / lệch bên, §16): 25% (1/4)
  type 85%  position 85%  | 836 ms  10.3 tok  sai schema 0
  cùng-1-cây-cột: gần=pole|front|near|move_left xa=pole|front|near|move_left lệch-bên=pole|left|near|move_right
                  -> KHÔNG phân biệt được xa/gần (cả hai đều near)
    CLR clear_00.jpg               object|front_left|near|move_left   BÁO THỪA
    CLR clear_01.jpg               vehicle|front_right|mid|slow       ok
    HAZ clear_02.jpg               person|front|mid|slow              BỎ SÓT
    CLR clearpath_00.jpg           clear|none|none|none               ok
    HAZ clearpath_01.jpg           object|front|near|stop             ok
    HAZ crowd_00.jpg               person|left|near|none              ok
    HAZ crowd_01.jpg               person|front|near|slow             ok
    HAZ door_00.jpg                door|front|near|slow               ok
    HAZ door_01.jpg                door|front|near|move_left          ok
    HAZ lowlight_00.jpg            person|front_left|near|move_left   ok
    HAZ lowlight_01.jpg            person|front|near|slow             ok
    HAZ pole_00.jpg                pole|left|near|none                ok
    HAZ street_00.jpg              person|front_left|near|slow        ok
    CLR street_01.jpg              clear|none|none|none               ok
    HAZ synth_person_front.jpg     person|front|near|slow             ok
    HAZ synth_pole_right.jpg       pole|right|near|move_left          ok
    CLR synthclear_dark.jpg        clear|none|none|none               ok
    CLR synthclear_grey.jpg        clear|none|none|none               ok
    CLR synthclear_indoor.jpg      clear|none|none|none               ok
    CLR synthclear_tiled.jpg       clear|none|none|none               ok
    CLR synthclear_warm.jpg        clear|none|none|none               ok
    CLR synthfar_person.jpg        object|front|mid|none              ok
    CLR synthfar_pole.jpg          pole|front|near|move_left          BÁO THỪA
    HAZ synthnear_pole.jpg         pole|front|near|move_left          ok
    CLR synthside_pole.jpg         pole|left|near|move_right          BÁO THỪA
    HAZ vehicle_00.jpg             object|left|mid|move_left          BỎ SÓT
    HAZ vehicle_01.jpg             object|front_left|near|move_left   ok

=== Tổng hợp (sắp theo balanced accuracy) ===
prompt      recall  specif  sp.khó  balanc   type    pos      ms    tok
pipe_v6       87%    75%    25%    81%   85%   85%    836   10.3
pipe_v5       73%    83%    50%    78%   73%   73%    714    9.6
pipe_v1       87%    67%    50%    77%   77%   77%    646    7.9
pipe_v3       60%    83%    50%    72%   78%   89%    605    7.2
pipe_v4       53%    67%    25%    60%   75%   88%    600    7.0
pipe_v2       20%    92%    75%    56%  100%   67%    568    6.4

Tốt nhất: pipe_v6 — balanced 81%, recall 87%, specificity 75% (âm khó 25%), 836 ms
  CẢNH BÁO: recall 87% nghĩa là cứ 10 vật cản thì bỏ sót 1. Với thiết bị dẫn đường thì KHÔNG dùng được một mình — §16 cần thêm detector/depth, hoặc đổi model lớn hơn.
  CẢNH BÁO: specificity trên frame âm khó chỉ 25% — model báo cả vật ở xa và vật lệch hẳn sang bên. Đúng nguy cơ 'overload thông tin' mà §30 lo. Decision Engine (§14/§16) phải tự lọc theo khoảng cách, không tin được VLM ở khoản này.

LƯU Ý: chỉ 12 frame ground-truth âm nên specificity có bước nhảy 8%. §8 vẫn cần 100-500 frame thật có nhãn.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/713 [00:00<?, ?it/s]
Loading weights:   0%|          | 1/713 [00:21<4:11:33, 21.20s/it]
Loading weights:  17%|█▋        | 121/713 [00:21<01:13,  8.05it/s]
Loading weights:  28%|██▊       | 198/713 [00:31<01:04,  7.96it/s]
Loading weights:  32%|███▏      | 231/713 [00:35<01:00,  8.01it/s]
Loading weights:  35%|███▍      | 249/713 [00:37<00:56,  8.19it/s]
Loading weights:  37%|███▋      | 261/713 [00:38<00:55,  8.17it/s]
Loading weights:  38%|███▊      | 269/713 [00:40<00:57,  7.78it/s]
Loading weights:  39%|███▉      | 278/713 [00:41<00:57,  7.59it/s]
Loading weights:  41%|████      | 289/713 [00:42<00:55,  7.61it/s]
Loading weights:  42%|████▏     | 300/713 [00:44<00:55,  7.46it/s]
Loading weights:  44%|████▎     | 311/713 [00:45<00:53,  7.56it/s]
Loading weights:  45%|████▍     | 320/713 [00:45<00:42,  9.29it/s]
Loading weights:  45%|████▌     | 323/713 [00:47<00:54,  7.13it/s]
Loading weights:  47%|████▋     | 333/713 [00:48<00:54,  6.98it/s]
Loading weights:  48%|████▊     | 344/713 [00:50<00:52,  7.07it/s]
Loading weights:  50%|████▉     | 355/713 [00:51<00:49,  7.22it/s]
Loading weights:  53%|█████▎    | 377/713 [00:51<00:25, 13.29it/s]
Loading weights:  54%|█████▍    | 388/713 [00:52<00:19, 17.04it/s]
Loading weights:  68%|██████▊   | 486/713 [00:52<00:03, 68.16it/s]
Loading weights:  87%|████████▋ | 617/713 [00:52<00:00, 155.38it/s]
Loading weights:  96%|█████████▋| 688/713 [00:52<00:00, 202.22it/s]
Loading weights: 100%|██████████| 713/713 [00:52<00:00, 13.61it/s]
```

### depth_probe_outdoor

- Exit code: `0`
- Duration: `9.012 s`

```text
Depth : depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
Load  : 0.7s | VRAM 48 MiB | fp16
Frames: 27 (15 có vật cản / 12 trống) | 640px | p10 | mét

--- Latency depth (27 lần) ---
  median 10 ms | min/max 10/103 ms   (VLM 4B đang là ~845 ms)

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
Kết quả thô -> benchmarks/depth_outdoor.json
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/287 [00:00<?, ?it/s]
Loading weights:  44%|████▎     | 125/287 [00:00<00:00, 1230.13it/s]
Loading weights: 100%|██████████| 287/287 [00:00<00:00, 1419.58it/s]
Loading weights: 100%|██████████| 287/287 [00:00<00:00, 1389.87it/s]
```

### depth_probe_indoor

- Exit code: `0`
- Duration: `8.7 s`

```text
Depth : depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
Load  : 0.7s | VRAM 48 MiB | fp16
Frames: 27 (15 có vật cản / 12 trống) | 640px | p10 | mét

--- Latency depth (27 lần) ---
  median 10 ms | min/max 10/86 ms   (VLM 4B đang là ~845 ms)

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
Kết quả thô -> benchmarks/depth_indoor.json
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/287 [00:00<?, ?it/s]
Loading weights:  44%|████▎     | 125/287 [00:00<00:00, 1241.51it/s]
Loading weights: 100%|██████████| 287/287 [00:00<00:00, 1438.30it/s]
Loading weights: 100%|██████████| 287/287 [00:00<00:00, 1407.78it/s]
```

