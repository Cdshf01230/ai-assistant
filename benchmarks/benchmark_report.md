# Benchmark report

- Mode: `inventory`
- Started: `2026-08-24 08:10:01 UTC`

## Dataset checks

| Dataset | Status | Details |
|---|---|---|
| `diode` | `ok` | size_bytes=9173067334, depth_arrays=1542, rgb_images=771 |
| `nyu_depth_v2` | `ok` | size_bytes=2972037809, keys=['#refs#', '#subsystem#', 'accelData', 'depths', 'images', 'instances', 'labels', 'names', 'namesToIds', 'rawDepthFilenames', 'rawDepths', 'rawRgbFilenames', 'sceneTypes', 'scenes'], missing_keys=[], images_shape=[1449, 3, 640, 480], depths_shape=[1449, 640, 480] |
| `sun_rgbd` | `ok` | size_bytes=7174366246, files=285925 |
| `coco` | `ok` | size_bytes=1087323831, annotation_files=['annotations/captions_train2017.json', 'annotations/captions_val2017.json', 'annotations/instances_train2017.json', 'annotations/instances_val2017.json', 'annotations/person_keypoints_train2017.json', 'annotations/person_keypoints_val2017.json'] |
| `fleurs_vi` | `ok` | size_bytes=3061201006, cached_files=9, validation_rows=361 |
| `ai2thor` | `ok` | package=ai2thor |
