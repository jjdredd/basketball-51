# CPU Inference Results — Basketball_51

All 24 videos processed with `infer.py --cpu` using the best checkpoint
(`best_acc_top1_epoch_41_fixed.pth`).

## Per-video results

| Video | Ground Truth | Prediction | Top-1 Conf | Top-2 | Correct? |
| --- | --- | --- | --- | --- | --- |
| 2p0_v108_000437_x264.mp4 | 2-pt miss | Misc miss | 30.7% | 2-pt miss (26.3%) | ✗ |
| 2p0_v153_003129_x264.mp4 | 2-pt miss | 2-pt miss | 60.1% | 3-pt miss (21.9%) | ✓ |
| 2p0_v184_012849_x264.mp4 | 2-pt miss | 2-pt miss | 75.6% | 2-pt made (24.1%) | ✓ |
| 2p1_v108_000340_x264.mp4 | 2-pt made | 2-pt made | 93.3% | 2-pt miss (6.2%) | ✓ |
| 2p1_v153_002725_x264.mp4 | 2-pt made | 2-pt made | 78.9% | 2-pt miss (18.1%) | ✓ |
| 2p1_v184_013447_x264.mp4 | 2-pt made | 2-pt made | 93.6% | 3-pt made (5.4%) | ✓ |
| 3p0_v108_000322_x264.mp4 | 3-pt miss | 3-pt miss | 55.8% | Misc miss (28.0%) | ✓ |
| 3p0_v150_003112_x264.mp4 | 3-pt miss | 3-pt miss | 74.5% | 3-pt made (17.2%) | ✓ |
| 3p0_v184_012835_x264.mp4 | 3-pt miss | 3-pt miss | 98.5% | 3-pt made (0.7%) | ✓ |
| 3p1_v108_001141_x264.mp4 | 3-pt made | 3-pt miss | 64.9% | Misc miss (15.0%) | ✗ |
| 3p1_v152_001113_x264.mp4 | 3-pt made | 3-pt made | 65.7% | 3-pt miss (26.4%) | ✓ |
| 3p1_v184_012410_x264.mp4 | 3-pt made | 3-pt made | 94.9% | 2-pt made (3.3%) | ✓ |
| ft0_v108_002649_x264.mp4 | FT miss | FT made | 83.7% | FT miss (16.3%) | ✗ |
| ft0_v156_012735_x264.mp4 | FT miss | FT miss | 74.2% | FT made (25.7%) | ✓ |
| ft0_v184_013102_x264.mp4 | FT miss | FT miss | 94.6% | FT made (5.4%) | ✓ |
| ft1_v108_000302_x264.mp4 | FT made | FT made | 99.5% | FT miss (0.5%) | ✓ |
| ft1_v150_001437_x264.mp4 | FT made | FT made | 85.6% | FT miss (14.4%) | ✓ |
| ft1_v184_013433_x264.mp4 | FT made | FT made | 100.0% | FT miss (0.0%) | ✓ |
| mp0_v108_000245_x264.mp4 | Misc miss | Misc miss | 74.5% | Misc made (16.7%) | ✓ |
| mp0_v145_010308_x264.mp4 | Misc miss | Misc miss | 27.0% | 2-pt miss (26.8%) | ✓ |
| mp0_v184_005922_x264.mp4 | Misc miss | 2-pt miss | 81.7% | 2-pt made (9.8%) | ✗ |
| mp1_v108_000416_x264.mp4 | Misc made | Misc made | 43.8% | 2-pt made (24.7%) | ✓ |
| mp1_v133_001745_x264.mp4 | Misc made | Misc made | 75.8% | 3-pt made (23.9%) | ✓ |
| mp1_v183_013613_x264.mp4 | Misc made | 3-pt made | 65.5% | 2-pt made (27.9%) | ✗ |

## Per-class accuracy

| Class | Accuracy |
| --- | --- |
| **2-pt miss** | 2/3 = 66.7% |
| **2-pt made** | 3/3 = 100.0% |
| **3-pt miss** | 3/3 = 100.0% |
| **3-pt made** | 2/3 = 66.7% |
| **FT miss** | 2/3 = 66.7% |
| **FT made** | 3/3 = 100.0% |
| **Misc miss** | 2/3 = 66.7% |
| **Misc made** | 2/3 = 66.7% |
| **Overall** | **19/24 = 79.2%** |

## Observations

- **Strong classes** (100%): 2-pt made, 3-pt miss, FT made — distinctive motions
  (free-throw form, made swish, 3-pt arc miss).
- **Confusable classes** (66.7%): miss vs made within the same shot type gets
  confused, especially for FT miss (confused with FT made on one sample) and
  3-pt made (confused with 3-pt miss on one sample).
- **Worst errors**:
  - `2p0_v108_000437`: 2-pt miss → Misc miss (only 30.7% confidence — very unsure)
  - `ft0_v108_002649`: FT miss → FT made (83.7% confident in wrong direction)
  - `mp0_v184_005922`: Misc miss → 2-pt miss (81.7% — confident but wrong class)
- **Top-2 rescue**: In 4 of 5 errors, the true class appears in the top-2 prediction,
  suggesting a top-3 or top-4 evaluation would push accuracy toward 90%+.
