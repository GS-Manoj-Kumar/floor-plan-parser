# Review of existing floor-plan recognition repos

Six repositories were reviewed before building `fpvec`. Each was cloned and its code read.

| Repo | Approach | Outputs | Needs model weights? | Usable as-is for raster → vector walls/doors/windows? |
|---|---|---|---|---|
| [krevas/FloorPlanRecognition](https://github.com/krevas/FloorPlanRecognition) | Fork of **DeepFloorplan** (ICCV 2019). Multi-task VGG network → per-pixel masks for room boundary (wall / door-window) and room type | Segmentation PNGs; no vectors | Yes (pretrained R3D checkpoint, TF1) | No. Python 2.7 / TensorFlow 1.10. Masks are not vectors, and doors and windows share one class |
| [pimenoffd/floor-plan-vectorizer](https://github.com/pimenoffd/floor-plan-vectorizer) | PyTorch segmentation model trained on CubiCasa5k, then `cv2.approxPolyDP` on the masks | JSON polygons for walls and openings | Yes, and you train them yourself (no weights shipped) | Partly. Good JSON idea, but polygons are not wall centrelines with thickness, and it needs a GPU training run first |
| [akramsystems/Wall-Detection](https://github.com/akramsystems/Wall-Detection) | CubiCasa5k "floortrans" hourglass network (Kalervo et al.) behind a FastAPI. Detects wall **corners / junction heatmaps** | Corner points + confidence | Yes (`model_best_val_loss_var.pkl`, not included) | No. Gives junction points only, with no wall segments, doors or windows |
| [Maxi-online/floor-plan-recognition](https://github.com/Maxi-online/floor-plan-recognition) | **Classical CV**: CLAHE + bilateral + Otsu/adaptive threshold → `HoughLinesP` → snap to axis → merge collinear. Rooms via SAM 2.1 / OCR (optional) | Wall line segments JSON | No for walls (SAM is optional) | Closest in spirit. But Hough on raw ink picks up furniture and text, there's no wall thickness, and no door/window detection |
| [pjm4github/FloorPlanner](https://github.com/pjm4github/FloorPlanner) | A full PyQt6 floor-plan **editor** (walls, openings, rooms, 3D, DXF/PDF export) | Its own design JSON schema (v6) | n/a | Not a recognizer. Its data model (walls with hosted openings, offset along wall) is a good target format, and `fpvec` output maps onto it directly |
| [TINY-KE/FloorPlanParser](https://github.com/TINY-KE/FloorPlanParser) | Closed-source paid web service; the repo is just a Python client + demo images | JSON start/end points for walls, doors, windows, bay windows | n/a (remote) | No. Needs an account, 10 calls/day demo, no algorithm included |

## What `fpvec` takes from them

* **DeepFloorplan / CubiCasa**: wall thickness and openings belong to the *wall*. Output walls as centrelines + thickness with hosted openings, not raw polygons.
* **Maxi-online**: classical CV is enough for clean CAD/sales plans and needs no GPU or weights. But `fpvec` replaces Hough with **morphology** (stroke-thickness filtering), which separates walls from furniture far better.
* **pimenoffd / FloorPlanParser**: the JSON shape (start/end points per element, px and mm).
* **FloorPlanner**: the hosted-opening model (`host_wall_ids`, `offset_on_host_mm`).

## When to add a learned model

`fpvec` is deterministic and has no dependencies beyond OpenCV. Where it struggles (scanned/photographed plans, hand drawings, heavily decorated marketing plans), the best upgrade is to replace step 1 (`wall_mask`) with a segmentation network trained on CubiCasa5k (as in pimenoffd or akramsystems). The rest of the pipeline (segment extraction, gap finding, arc-based door detection, exports) stays the same: it only needs a binary wall mask.
