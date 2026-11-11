# Preprocessing optimization

Date: 2026-10-15. Scope: the in-memory preprocessing pipeline
(`BackgroundEraser -> Normalizer -> Repair -> Vectorizer`, config
`configs/prep_vector_axes_svg.yaml`). Feature learning has not been profiled
or changed yet.

## Result

Measured on a random sample of 188 raw legion scans (2480 x 3507 px, grayscale
JPEG), memory mode, batch size 64, on one free RTX 3090 and a 36-core CPU:

| Step | Before | After |
|---|---|---|
| Image loading | 2.9 s | 0.6 s |
| BackgroundEraser | 36.2 s | 4.5 s |
| Normalizer | 0.5 s | 0.6 s |
| Repair | 0.1 s | 0.1 s |
| Vectorizer (with SVG) | 2.1 s | 2.1 s |
| **Total** | **42.0 s** | **7.7 s** (about 5.5x faster) |

For the whole `/data/alaczko/legion_raw` (6277 images), that is roughly 4–5 minutes
instead of about 24.

**The output is unchanged.** A copy of the old code was run on the same sample,
and its output was compared with the new code's. All 186 final images, `axes.csv`
and all 186 SVGs were identical byte for byte. The BackgroundEraser output and
axes tables were also identical in both `segmentation` and `detection` mode.

## Where the time went

Before the change, 87% of the time was spent in `BackgroundEraser`. YOLO itself
took only about 6 s of that. The rest was `_clear_one`, running on one CPU
core:

| Cost | Time |
|---|---|
| Axis search (`detect_axes`) | 10.2 s |
| Mask rasterizing, whitening (`np.where`, `full_like`) and cropping on the full 3-channel A4 scan | ~10 s |
| `np.nonzero` on full-resolution masks, twice per image, only to get a bounding box | 8.4 s |
| Mask dilation on the full scan | 4.5 s |

## Changes

All of these are in `pipeline/preprocessing/background_eraser.py` unless noted
otherwise.

1. **Parallel per-image work.** After YOLO, a batch's images are cleared on a
   thread pool. This covers whitening, cropping and, with `record_axes`, the
   axis search. OpenCV and numpy release the GIL, so threads are enough.
   `_clear_one` no longer changes `self`: it returns the axes row, and
   `_store_axes` records the rows in input order. Results therefore do not
   depend on the number of threads. It is set by the new parameter
   `num_workers`, with these values:
   - `None` (default): `min(16, CPU count)`
   - `0` or `1`: serial

   Disk mode (`_clear_images`) uses the same pool, for reading, clearing and
   writing.

2. **No `np.nonzero`.** The crop is now computed from the polygon's bounding
   rectangle (`cv2.boundingRect`, clipped to the image) plus the dilation
   radius, before any mask is drawn. `_dilate_mask` was replaced by
   `_dilate_radius(width, height, margin)`.

3. **Work only around the crop square.** The mask is rasterized
   (`cv2.fillPoly` with an offset), dilated and applied only inside the crop
   square, plus a border the size of the dilation radius. It is no longer done
   on the whole A4 sheet. The border means the dilation sees every mask pixel
   that reaches into the square, so the result is exactly the same.
   `_save_debug_image` now takes the region's offset so the overlay is still
   drawn in full-sheet coordinates.

4. **Prefetch and parallel loading** (`pipeline/preprocessing/dataset.py`,
   `pipeline/pipeline.py`).
   - `iter_image_batches` decodes a batch on `num_workers=8` threads.
   - The new `prefetch(iterable, depth=1)` loads the next batch in a
     background thread while the current one goes through the steps. Errors
     in that thread are raised in the caller.
   - `Pipeline.iter_batches` uses both.

5. **Grayscale stays one channel.** Grayscale scans (PIL mode `L`, or
   `cv2.IMREAD_ANYCOLOR` in disk mode) are processed as `(H, W)` arrays
   instead of being converted to three identical channels. That is a third of
   the memory work, with the same pixels. Only YOLO gets a 3-channel copy.
   Color images stay BGR/RGB as before.

6. **Optional FP16 YOLO (`half`).** This was added as an option, but it is
   **off by default and not recommended**. On the sample it gave no speed-up
   (8.1 s vs 7.7 s), and 75 of 186 final images changed slightly because the
   masks differ.

## Configuration

Two new `BackgroundEraser` options (also listed in `configs/template.yaml`):

```yaml
    - BackgroundEraser:
        num_workers: null    # threads clearing a batch in parallel; null = min(16, CPUs), 1 = one at a time
        half: false          # true: FP16 YOLO (faster on GPU; masks may differ by a pixel from FP32)
```

Existing configs need no changes.

## Tests

New tests:

- `tests/test_background_eraser_axes.py::test_crop_only_mask_matches_whole_sheet_mask`:
  the crop-only mask gives the same pixels as rasterizing, dilating and
  whitening the whole sheet.
- `tests/test_background_eraser_axes.py::test_parallel_and_grayscale_give_the_same_result`:
  serial, parallel and grayscale runs give the same images, axes tables and
  `no_axes` lists.
- `tests/test_pipeline.py::test_prefetch_keeps_order_and_raises_errors`.

Full suite: 105 passed, 2 failed. The two failures were already failing before
these changes:

- `test_early_stopping::test_vae_train_model_stops_early_when_loss_plateaus`
- `test_feature_learning::TestFeatureExport::test_extract_features_saves_filename_and_vector_csv`

## Possible next steps

- **Vectorizer** is now the largest preprocessing step (2.1 s per 188 images).
  Its skeleton tracing (`pixel_neighbors`, `dijkstra_farthest`) is pure Python
  and could be parallelized the same way.
- **Training (not measured yet):**
  - `vae.py` calls `.item()` four times per batch. Each call waits for the
    GPU, so loading and computation never overlap. Summing on the GPU and
    reading the value once per epoch would avoid this.
  - In memory mode, the whole dataset (about 400 MB as uint8) fits on the
    GPU. Without augmentation, this would make the 16-worker DataLoader and
    the per-image PIL conversion unnecessary.
  - Mixed precision (`torch.autocast`) is another option, with care for the
    BCE loss.
  - The first step should be to time one epoch.
