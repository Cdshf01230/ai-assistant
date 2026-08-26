# VLM+depth trên COCO val2017 (nhãn từ GT bbox)

```
{
  "n": 120,
  "n_positive": 60,
  "depth_threshold_m": 7.0,
  "latency_ms": {
    "vlm_median": 882,
    "vlm_p95": 1076,
    "depth_median": 77
  },
  "A_vlm_near_mid": {
    "recall": 0.8,
    "specificity": 0.4167,
    "balanced": 0.6083,
    "tp": 48,
    "fp": 35,
    "fn": 12,
    "tn": 25
  },
  "B_vlm_near_only": {
    "recall": 0.7167,
    "specificity": 0.4333,
    "balanced": 0.575,
    "tp": 43,
    "fp": 34,
    "fn": 17,
    "tn": 26
  },
  "E_fusion_calibrated": {
    "recall": 0.7333,
    "specificity": 0.5167,
    "balanced": 0.625,
    "tp": 44,
    "fp": 29,
    "fn": 16,
    "tn": 31
  }
}
```
