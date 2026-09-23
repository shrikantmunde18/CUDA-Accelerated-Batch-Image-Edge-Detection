# Batch CUDA Image Edge Detection Pipeline

**GPU Specialization — Capstone Project**

## Overview

This project implements a GPU-accelerated image processing pipeline in raw CUDA C++
(no Thrust/NPP shortcuts — hand-written kernels) that batch-processes an entire
directory of images through three sequential stages, all executed on the device:

1. **RGB → Grayscale conversion** — a simple per-pixel luminance-weighted kernel.
2. **5×5 Gaussian Blur** — noise reduction using a fixed, normalized 5×5 kernel
   stored in `__constant__` memory for fast broadcast reads across threads.
3. **3×3 Sobel Edge Detection** — computes horizontal/vertical gradients (Gx, Gy)
   per pixel, combines them into a gradient magnitude, and thresholds the result
   to produce a binary-ish edge map.

Each image is loaded on the host with [stb_image](https://github.com/nothings/stb)
(single-header, public-domain image I/O), copied to device memory, run through the
three kernels back-to-back without intermediate host round-trips, copied back, and
written out as a PNG with `stb_image_write`. Per-image GPU execution time is measured
with CUDA events and logged to a CSV so throughput across the whole batch can be
analyzed afterward.

This was built as an extension of the Thrust-based radix sort lab from this course —
instead of relying on a library algorithm, this project hand-writes the CUDA kernels
and manages device memory directly, to demonstrate the lower-level GPU programming
model (grid/block indexing, constant memory, boundary clamping, event-based timing).

## Why this project

Image edge detection is a classic "many large pieces of data" GPU workload — each
image is a big 2D array of independent pixels, which maps naturally onto CUDA's
`(blockIdx, threadIdx)` grid model with essentially zero inter-thread communication
needed within a stage. Running it as a *batch* (a whole directory of images in a
single execution) was a deliberate choice to make GPU utilization and total
throughput easy to observe and measure, rather than showing a single one-off run.

## Repository Layout

```
.
├── Makefile               # Build rules (nvcc), matches course lab conventions
├── run.sh                 # One-command build + run + log capture
├── README.md               # This file
├── src/
│   ├── edge_detect.cu      # All CUDA kernels + host pipeline code
│   ├── stb_image.h         # Public-domain image loader (single header)
│   └── stb_image_write.h   # Public-domain image writer (single header)
├── images_in/              # Sample input images (proof-of-execution dataset)
│   ├── lena.jpg
│   ├── baboon.jpg
│   └── fruits.jpg
└── images_out/              # Generated at run time: edge maps + timing_log.csv + run_log.txt
```

## Requirements

- NVIDIA CUDA Toolkit (developed/tested against the CUDA version provided in the
  Coursera GPU Specialization lab environment; any CUDA 10+ toolkit should work)
- A CUDA-capable GPU (compute capability 5.0+; the Makefile's `SMS` list covers
  common architectures from Maxwell through Ampere — override with
  `make SMS="75" build` if your target GPU isn't in the default list)
- `g++` (used as the host compiler by `nvcc`)
- No external libraries required beyond the CUDA Toolkit — `stb_image.h` and
  `stb_image_write.h` are vendored directly in `src/` so there is nothing extra
  to install.

## Build

```bash
make build
```

This compiles `src/edge_detect.cu` into `edge_detect.exe`.

To remove build artifacts:

```bash
make clean
```

## Run

### Quickest path (recommended)

```bash
./run.sh
```

This cleans, rebuilds, runs the pipeline over every image in `images_in/`, and
writes results + a full console log into `images_out/`.

### Manual invocation / CLI arguments

```bash
./edge_detect.exe <input_dir> <output_dir> [--threshold N]
```

| Argument      | Required | Description                                                        |
|---------------|----------|----------------------------------------------------------------------|
| `input_dir`   | Yes      | Directory containing `.png`/`.jpg`/`.jpeg`/`.bmp` images to process   |
| `output_dir`  | Yes      | Directory to write `<name>_edges.png` outputs + `timing_log.csv`     |
| `--threshold` | No       | Integer Sobel gradient-magnitude threshold, 0–255 (default: `100`)   |

Example, running on a different dataset with a custom threshold:

```bash
./edge_detect.exe /path/to/my_photos ./results --threshold 60
```

You can also run via Make with the default sample dataset:

```bash
make run
```

## Output / Proof of Execution

After a run, `images_out/` contains:

- `<name>_edges.png` — the edge-map output for every input image
- `timing_log.csv` — per-image GPU execution time in milliseconds (`image,gpu_time_ms`)
- `run_log.txt` — full console output of the run (device name detected, per-image
  processing lines, and a final summary of total/average GPU time across the batch)

This is included in the repository already populated from a sample run over three
512×512-class test images (`lena.jpg`, `baboon.jpg`, `fruits.jpg`), so the effect of
the pipeline and its measured performance are visible without needing to re-run
anything — though re-running with `./run.sh` is straightforward and will regenerate
everything from scratch.

## Algorithm / Implementation Notes

- **Grid/block configuration:** all three kernels use 16×16 thread blocks with a
  grid sized to cover the image in both dimensions (`(width + 15) / 16` blocks
  wide, same for height), with an explicit bounds check inside each kernel so
  images whose dimensions aren't multiples of 16 are still handled correctly.
- **Boundary handling:** the blur and Sobel kernels clamp out-of-bounds neighbor
  reads to the nearest valid edge pixel, rather than reading OOB or wrapping.
- **Constant memory:** the 5×5 Gaussian weights are stored in `__constant__`
  memory, which is cached and broadcast-efficient since every thread in a warp
  reads the same 25 values.
- **Timing:** each image's three-kernel pipeline (grayscale → blur → Sobel) is
  timed as one block using `cudaEvent` start/stop around the three kernel
  launches, capturing pure GPU compute time (excludes host-side image I/O).
- **No Thrust used here on purpose** — this project intentionally uses raw
  kernels and manual `cudaMalloc`/`cudaMemcpy` to demonstrate direct GPU memory
  and execution-model management, as a complement to the Thrust-based work
  done earlier in the course.

## Lessons Learned / Challenges

- Chaining three kernels back-to-back on the same device buffers (grayscale →
  blur → Sobel) without copying back to host in between meaningfully reduces
  PCIe transfer overhead compared to a naive per-stage host round-trip —
  something that's easy to state in theory but is much clearer once you can
  see it in the per-image timing numbers.
- Boundary/edge-pixel handling is an easy source of subtle bugs in stencil-style
  kernels (blur, Sobel) — clamping neighbor coordinates to `[0, width-1]` /
  `[0, height-1]` instead of leaving them unchecked was necessary to avoid
  reading out-of-bounds device memory at image edges.
- Using `__constant__` memory for the fixed Gaussian kernel weights was a
  natural fit here since every thread reads the identical 25 values — a good
  concrete example of when constant memory's broadcast caching actually pays
  off versus just reading from global memory.

## Next Steps (if extended further)

- Add a separable Gaussian blur (two 1D passes instead of one 5×5 2D pass) to
  reduce the per-pixel work from 25 multiply-adds to 10.
- Use shared memory tiling for the blur/Sobel stencils to cut down on redundant
  global memory reads at block boundaries.
- Add non-maximum suppression + double-thresholding to turn the current Sobel
  magnitude map into a full Canny edge detector.
