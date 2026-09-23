#!/usr/bin/env bash
# Builds the project, then runs the GPU pipeline over every image in
# images_in/, writing edge-detected outputs and a timing_log.csv into
# images_out/. Also captures full stdout for proof-of-execution.
set -e

make clean
make build

mkdir -p images_out
./edge_detect.exe images_in images_out --threshold 100 | tee images_out/run_log.txt

echo ""
echo "=== Contents of images_out/ ==="
ls -la images_out
