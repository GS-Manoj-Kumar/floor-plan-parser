# Sample set results

15 generated coloured floor plans (5 drawing styles), each as a **clean** image and a **distorted** one
(a phone photo at an angle for odd numbers, a crooked yellowed scan for even numbers). Run with
`process(image, auto_scale=True, photo="auto", wall_mm=200)`, scored by `fpvec/evaluate.py` against the
generator's ground truth. Regenerate with `python samples/run_samples.py`.

| Set | Plans | Scale found automatically | Median size error | Wall recall | Wall precision | Walls, shape only (R / P) | Interior wall recall | Doors found | False doors | Windows found | False windows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 15 | 15 / 15 (11 from dimensions, 4 from room areas) | 0.12 % | 100% | 93% | 100% / 93% | 100% | 88% | 3 | 98% | 14 |
| distorted | 15 | 14 / 15 (11 from dimensions, 3 from room areas) | 0.28 % | 93% | 94% | 96% / 98% | 86% | 75% | 4 | 91% | 7 |

* **Wall recall / precision**: share of true wall length matched by a predicted wall within 150 mm, and vice versa,
  in millimetres with the scale fallback included, so a wrong scale lowers it.
* **Walls, shape only**: the same after fitting the plan's overall size, which shows detection quality without the scale.
* **Doors / windows**: centre within 350 mm of a true one of the same kind.
* **Scale**: dimension text first, then room-area labels ("38.4 m2") against the rooms the walls enclose,
  then a 200 mm main wall is assumed (the true exterior walls are 200 or 240 mm).

| Sample | Variant | Style | Dims | Scale | Size err | Walls R / P | Doors | Windows | Correction |
|---|---|---|---|---|---|---|---|---|---|
| [01_sales](output/01_sales_clean_overlay.jpg) | clean | sales | mm | dimensions (5) | 0.11 % | 100% / 100% | 3/4 (+1) | 3/3 (+0) | – |
| [01_sales](output/01_sales_photo_overlay.jpg) | photo | sales | mm | dimensions (5) | 0.58 % | 100% / 100% | 4/4 (+0) | 3/3 (+0) | perspective, lighting |
| [02_wood](output/02_wood_clean_overlay.jpg) | clean | wood | ftin | dimensions (6) | 0.03 % | 100% / 82% | 2/3 (+1) | 3/3 (+1) | – |
| [02_wood](output/02_wood_scan_overlay.jpg) | scan | wood | ftin | dimensions (4) | 0.24 % | 100% / 100% | 3/3 (+0) | 3/3 (+0) | rotate |
| [03_blue](output/03_blue_clean_overlay.jpg) | clean | blue | mm | dimensions (4) | 0.1 % | 100% / 100% | 3/4 (+0) | 3/3 (+0) | – |
| [03_blue](output/03_blue_photo_overlay.jpg) | photo | blue | mm | dimensions (5) | 1.6 % | 100% / 100% | 3/4 (+0) | 3/3 (+0) | perspective, lighting |
| [04_grey_partitions](output/04_grey_partitions_clean_overlay.jpg) | clean | grey_partitions | mm | dimensions (5) | 0.1 % | 100% / 99% | 3/3 (+0) | 4/4 (+1) | – |
| [04_grey_partitions](output/04_grey_partitions_scan_overlay.jpg) | scan | grey_partitions | mm | dimensions (5) | 0.11 % | 88% / 100% | 1/3 (+0) | 4/4 (+0) | rotate |
| [05_brown_walls](output/05_brown_walls_clean_overlay.jpg) | clean | brown_walls | ftin | dimensions (5) | 0.07 % | 100% / 95% | 4/4 (+0) | 2/3 (+2) | – |
| [05_brown_walls](output/05_brown_walls_photo_overlay.jpg) | photo | brown_walls | ftin | dimensions (3) | 1.03 % | 100% / 83% | 4/4 (+0) | 3/3 (+0) | perspective, lighting |
| [06_sales](output/06_sales_clean_overlay.jpg) | clean | sales | – | room areas (3) | 0.35 % | 100% / 99% | 3/4 (+0) | 4/4 (+1) | – |
| [06_sales](output/06_sales_scan_overlay.jpg) | scan | sales | – | room areas (4) | 0.03 % | 100% / 100% | 2/4 (+0) | 4/4 (+0) | rotate |
| [07_wood](output/07_wood_clean_overlay.jpg) | clean | wood | ftin | dimensions (5) | 0.15 % | 100% / 100% | 3/4 (+0) | 3/3 (+0) | – |
| [07_wood](output/07_wood_photo_overlay.jpg) | photo | wood | ftin | dimensions (2) | 1.03 % | 100% / 83% | 3/4 (+0) | 3/3 (+3) | perspective, lighting |
| [08_blue](output/08_blue_clean_overlay.jpg) | clean | blue | – | room areas (7) | 1.05 % | 100% / 100% | 7/7 (+0) | 3/3 (+0) | – |
| [08_blue](output/08_blue_scan_overlay.jpg) | scan | blue | – | room areas (5) | 0.74 % | 100% / 100% | 6/7 (+1) | 3/3 (+0) | rotate |
| [09_grey_partitions](output/09_grey_partitions_clean_overlay.jpg) | clean | grey_partitions | – | room areas (4) | 0.39 % | 99% / 100% | 3/4 (+0) | 4/4 (+0) | – |
| [09_grey_partitions](output/09_grey_partitions_photo_overlay.jpg) | photo | grey_partitions | – | wall (0) | 10.08 % | 31% / 41% | 0/4 (+2) | 0/4 (+4) | perspective, lighting |
| [10_brown_walls](output/10_brown_walls_clean_overlay.jpg) | clean | brown_walls | ftin | dimensions (4) | 0.22 % | 100% / 69% | 6/6 (+0) | 3/3 (+2) | – |
| [10_brown_walls](output/10_brown_walls_scan_overlay.jpg) | scan | brown_walls | ftin | dimensions (3) | 0.08 % | 100% / 98% | 6/6 (+0) | 3/3 (+0) | rotate |
| [11_sales](output/11_sales_clean_overlay.jpg) | clean | sales | mm | dimensions (7) | 0.1 % | 100% / 96% | 8/8 (+0) | 2/2 (+3) | – |
| [11_sales](output/11_sales_photo_overlay.jpg) | photo | sales | mm | dimensions (7) | 0.19 % | 100% / 100% | 8/8 (+0) | 2/2 (+0) | perspective, lighting, aspect |
| [12_wood](output/12_wood_clean_overlay.jpg) | clean | wood | ftin | dimensions (6) | 0.26 % | 100% / 67% | 4/5 (+1) | 1/1 (+3) | – |
| [12_wood](output/12_wood_scan_overlay.jpg) | scan | wood | ftin | dimensions (4) | 0.28 % | 100% / 100% | 2/5 (+1) | 1/1 (+0) | rotate |
| [13_blue](output/13_blue_clean_overlay.jpg) | clean | blue | – | room areas (4) | 0.78 % | 100% / 100% | 4/4 (+0) | 4/4 (+0) | – |
| [13_blue](output/13_blue_photo_overlay.jpg) | photo | blue | – | room areas (4) | 0.91 % | 100% / 100% | 4/4 (+0) | 4/4 (+0) | perspective, lighting |
| [14_grey_partitions](output/14_grey_partitions_clean_overlay.jpg) | clean | grey_partitions | mm | dimensions (5) | 0.12 % | 100% / 96% | 3/4 (+0) | 3/3 (+0) | – |
| [14_grey_partitions](output/14_grey_partitions_scan_overlay.jpg) | scan | grey_partitions | mm | dimensions (5) | 0.19 % | 77% / 100% | 1/4 (+0) | 3/3 (+0) | rotate |
| [15_brown_walls](output/15_brown_walls_clean_overlay.jpg) | clean | brown_walls | mm | dimensions (5) | 0.09 % | 100% / 98% | 4/4 (+0) | 3/3 (+1) | – |
| [15_brown_walls](output/15_brown_walls_photo_overlay.jpg) | photo | brown_walls | mm | dimensions (5) | 0.15 % | 100% / 100% | 4/4 (+0) | 3/3 (+0) | perspective, lighting, aspect |
