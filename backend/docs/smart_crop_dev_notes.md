# Smart Crop Development Notes

## Problem
When cropping portrait photos to 21:9 landscape format, the system needs to intelligently select which ~32% of the image height to show. The vision LLM (moondream, llava-phi3) provides focus points, subject bounding boxes, and person detection — but these are frequently hallucinated or inaccurate.

## Key Improvements (chronological)

### 1. Tiny Face Filter (image_captioner.py)
**Problem**: Haar cascade detects "faces" in clouds, rocks, tree textures (70-320px). These false positives trigger `has_person=True` → `mode=full` → bad crops focused on sky/textures.

**Fix**: Filter faces smaller than 3.5% of the image's smaller dimension. A 70px face in a 3024px image is filtered out (threshold=106px).

### 2. Proximity-Based Face Filter (smart_crop.py `_adjust_crop_for_faces`)
**Problem**: Multiple false face detections spread far apart (>25% image height) cause crop to follow wrong position. E.g., tree-texture "face" 800px above real child face.

**Fix**: When faces span >25% of image height, keep only face(s) closest to focus point (±10% tolerance). For tight clusters, keep faces within 25% of focus.

### 3. Single Small False Face Skip (smart_crop.py `_blur_fill`)
**Problem**: A single small false face (<10% image height) in the upper 40% triggers body estimation that shifts crop to sky-only.

**Fix**: When exactly one small face meets this pattern with default focus, skip all face logic entirely — treat as landscape.

### 4. LLM Caption Hallucination Filter (image_captioner.py)
**Problem**: Vision LLMs (moondream ~45%, llava-phi3 ~20%) parrot the example prompt response ("A woman walks along the beach at sunset") for images with no humans. This sets `has_person=True` and wrong focus/subject_box.

**Fix**: When **no CV detection** (no faces, no bodies), check caption text for human-related words. If found, replace caption with generic text and clear all LLM-derived positioning (focus, subject_box, has_person).

### 5. CV-Confirmed Person Detection (image_captioner.py)
**Problem**: LLM says `PERSON: YES` for spiral staircases, landscapes, etc. Without CV confirmation, `mode=full` is unreliable.

**Fix**: `has_person = bool(face_regions) or bool(body_rects)` — require at least one CV detector to confirm. LLM-only person claims are ignored.

### 6. Mode Override Without Face Confirmation (smart_crop.py `smart_fit`)
**Problem**: Cached captions with hallucinated `mode=full` still reach `_blur_fill`, which trusts LLM focus/subject_box → sky-only or legs-only crops.

**Fix**: In `smart_fit`, when `mode=full` but `face_regions` is empty, fall back to `_crop_fill` with default focus (0.5, 0.5) → pure saliency centering. All LLM data (focus, subject_box) is discarded as unreliable without CV confirmation.

### 7. Saliency Center Improvements (smart_crop.py `_saliency_center`)
**Problem**: Without horizon, saliency peak follows edge textures (grass, rocks) → extreme crop positions.

**Fix**: When no clear horizon detected, use vertical center (h/2) instead of saliency peak. Added 25%-75% sanity clamp on all results.

### 8. Horizon-Aware Cropping (smart_crop.py `_saliency_center`)
**Problem**: Landscape photos with clear horizon get cropped to sky-only.

**Fix**: Color-based boundary detection finds horizon/treeline. When detected in upper portion, shift crop center below it to show both horizon and ground content.

### 9. Max Face Size Cap (smart_crop.py `_filter_faces`)
**Problem**: Haar cascade detects entire cloud formations or tree canopy regions as giant "faces" (e.g., 871px in a 3024px image = 29%). These pass the minimum size filter and trigger face-aware cropping → shifts crop to sky.

**Fix**: Cap maximum face size at 20% of the image's smaller dimension. Rejects cloud-scale false positives while keeping real faces (typically 5-15% of image).

### 10. Face Adjustment Shift Limit (smart_crop.py `_adjust_crop_for_faces_limited`)
**Problem**: Even after filtering, remaining false faces in the sky area (e.g., 123px detections at y=1046) cause `_adjust_crop_for_faces` to shift the crop window far upward (sometimes >100% of crop height), producing sky-only results.

**Fix**: Wrapper function limits vertical shift from face adjustment to 40% of crop height. The initial crop position from body estimation or focus is usually reasonable; face adjustment refines it rather than overriding it.

### 11. Subject Box Minimum Area Check (smart_crop.py `_crop_fill`, `_blur_fill`)
**Problem**: LLM sometimes generates tiny subject_box coordinates pointing at sky (e.g., 25%×10% = 2.5% area at y=0.3). `_subject_box_crop` zooms into this tiny sky region.

**Fix**: Reject subject boxes with area < 5% or height < 15%. When rejected, also reset focus to default (0.5, 0.5) since the associated SUBJECT focus point from the same LLM call is equally unreliable. Saliency centering takes over.

### 12. Widened Single-Face Distance Tolerance (smart_crop.py `_adjust_crop_for_faces`)
**Problem**: Single false face at image edge (e.g., 191px at y=219 in 4032px image) should be rejected by the distance-from-focus check, but the check only fires when `abs(focus_y - 0.5) < 0.01`. LLM noise gives focus_y=0.475, bypassing the check.

**Fix**: Widened tolerance from 0.01 to 0.05 so "near-default" focus values still trigger the distance check.

### 13. Subject Box Upper-Third Focus for No-Face Persons (smart_crop.py `smart_fit`)
**Problem**: When mode=full + no faces + valid subject_box, the mode override discarded the box entirely and fell back to saliency centering (0.5, 0.5). For portrait photos of distant/small people (e.g., child on fence), saliency centers on water/scenery → legs-only crop.

**Fix**: When subject_box passes validation (area ≥5%, height ≥15%), use `sy1 + box_height * 0.33` as focus_y (upper-third of the box, biased toward head area). The partial-body heuristic (sy1 > 0.6, ratio > 3×) still handles lower-body-only boxes. This gives correct crops for children, distant walkers, and people photographed from low angles.

### 14. Proximity Filter — Centroid-Based with Relaxed Thresholds (smart_crop.py `_adjust_crop_for_faces`)
**Problem**: Proximity filter used LLM focus_y as reference, with a 25% spread threshold. For group photos with faces spanning just over 25%, upper faces were dropped because they were far from the LLM focus point (which pointed at one person).

**Fix**: Changed reference from LLM focus to face centroid (average of all face centers). Increased spread threshold from 25% to 35%, tolerance from 10% to 15%, and cluster threshold from 25% to 30%. Group photos now keep all real faces.

## Architecture Decisions

### Two-Mode System
- `mode=crop`: Scale-to-cover + saliency/focus centering. Used for landscapes, objects, scenes.
- `mode=full`: Person-aware with face-guided body estimation + head padding. Used when CV confirms faces.

### Trust Hierarchy
1. **Face regions** (Haar cascade) — most reliable for positioning
2. **Body detection** (HOG pedestrian) — confirms person presence
3. **LLM focus point** — useful when clamped, unreliable at extremes
4. **LLM subject_box** — only trusted when area ≥5% and height ≥15%; rejected boxes also invalidate focus
5. **Saliency center** — reliable fallback for unknown subjects

### Fundamental Limitations
- **Portrait-to-21:9**: Only ~32% of image height visible. Persons in top/bottom 20% will have heads/feet cut regardless of algorithm.
- **Small subjects**: Animals (hedgehogs), distant people — CV can't detect, LLM focus unreliable. Saliency centering is best-effort.
- **Close-up portraits**: Face too large for landscape crop. `check_face_fits()` detects and drops these.

### 15. LLM-Enhanced Captioning: ELEMENT, PEOPLE, HORIZON (image_captioner.py)
**Problem**: CV detectors (Haar, HOG) miss small/distant subjects. Saliency centering is unreliable for images with uniform textures (forest floor, sand). No horizon data available for landscape crops.

**Fix**: Enhanced LLM prompt to provide three new fields:
- `ELEMENT`: person/animal/landscape/architecture/object — classifies main subject
- `PEOPLE`: count + center positions of each person (more precise than subject_box)
- `HORIZON`: vertical position of horizon/skyline/treeline (or NONE for indoors)

Used in smart_fit: people_centers for mode=full+no_faces positioning, horizon_y for crop centering in _crop_fill, element_type for future use.

### 16. Hallucination Filter: Clear People + Element (image_captioner.py)
**Problem**: Hallucination filter cleared focus/subject_box when LLM mentions humans without CV confirmation, but left people_centers and element_type="person" intact. people_centers from hallucinated person claims are unreliable; element_type="person" contradicts the filter's correction.

**Fix**: Also clear `people_centers` and reset `element_type` to None when it's "person" in the hallucination filter. Keep `horizon_y` since horizons are physical landscape features, not person-dependent.

### 17. Horizon Logic: Center on Mid-Range Horizons (_crop_fill)
**Problem**: `_crop_fill` used `crop_cy = horizon_y * scaled_h + out_h * 0.20` for all LLM-provided horizons. This always shifts the crop BELOW the horizon. For beach scenes where scenic content (trees, hills) is ABOVE the horizon, this shows only sand.

**Fix**: Match `_saliency_center`'s approach:
- horizon < 45%: shift below to show boundary near crop top (typical upper-sky landscape)
- horizon 45-65%: center on it to show both above and below (safest for mid-range)
- horizon > 65%: ignore (fall through to saliency — likely false positive from ground texture)

**Effect on 061 (beach)**: With horizon=0.35, the upper-horizon path shifts below showing water, hills, trees, and sand — instead of sand-only.

## Model Notes
- **moondream**: Fast (1.7GB), ~45% hallucination rate, `/api/generate` only
- **llava-phi3**: Medium (2.9GB), ~20% hallucination, better descriptions, `/api/chat`
- **qwen2.5vl:7b**: Large (6.0GB), best accuracy, slower
- All vision LLMs have significant hallucination rates — CV confirmation is essential
