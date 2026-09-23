// edge_detect.cu
// GPU Specialization Capstone Project
// Batch CUDA Image Edge Detection Pipeline
//
// Pipeline per image: RGB -> Grayscale -> Gaussian Blur (5x5) -> Sobel Edge Detection
// All three stages run as custom CUDA kernels on the GPU.
//
// Usage:
//   ./edge_detect.exe <input_dir> <output_dir> [--threshold N]
//
// Reads every .png/.jpg/.bmp file in <input_dir>, runs the GPU pipeline on each,
// writes <name>_edges.png into <output_dir>, and appends per-image timing to
// <output_dir>/timing_log.csv

#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Kernel 1: RGB(A) -> Grayscale
// ---------------------------------------------------------------------------
__global__ void rgbToGrayscaleKernel(const unsigned char *rgb, unsigned char *gray,
                                      int width, int height, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int idx = (y * width + x) * channels;
  unsigned char r = rgb[idx];
  unsigned char g = rgb[idx + 1];
  unsigned char b = rgb[idx + 2];
  gray[y * width + x] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
}

// ---------------------------------------------------------------------------
// Kernel 2: 5x5 Gaussian Blur (separable-in-effect, done as a single pass here
// for simplicity/clarity; uses a fixed normalized kernel in constant memory)
// ---------------------------------------------------------------------------
__constant__ float d_gaussianKernel[25] = {
    1, 4, 6, 4, 1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4, 6, 4, 1
};
// Sum of the above weights = 256, used to normalize.

__global__ void gaussianBlurKernel(const unsigned char *in, unsigned char *out,
                                    int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float sum = 0.0f;
  int k = 0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      int sx = min(max(x + dx, 0), width - 1);
      int sy = min(max(y + dy, 0), height - 1);
      sum += in[sy * width + sx] * d_gaussianKernel[k++];
    }
  }
  out[y * width + x] = static_cast<unsigned char>(sum / 256.0f);
}

// ---------------------------------------------------------------------------
// Kernel 3: Sobel Edge Detection (3x3 Gx/Gy, magnitude output)
// ---------------------------------------------------------------------------
__global__ void sobelKernel(const unsigned char *in, unsigned char *out,
                             int width, int height, int threshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  int sumX = 0, sumY = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int sx = min(max(x + dx, 0), width - 1);
      int sy = min(max(y + dy, 0), height - 1);
      unsigned char px = in[sy * width + sx];
      sumX += Gx[dy + 1][dx + 1] * px;
      sumY += Gy[dy + 1][dx + 1] * px;
    }
  }

  int magnitude = static_cast<int>(sqrtf(static_cast<float>(sumX * sumX + sumY * sumY)));
  magnitude = min(magnitude, 255);
  out[y * width + x] = (magnitude >= threshold) ? static_cast<unsigned char>(magnitude) : 0;
}

// ---------------------------------------------------------------------------
// Host-side pipeline for a single image. Returns milliseconds of GPU time.
// ---------------------------------------------------------------------------
float processImage(const std::string &inPath, const std::string &outPath, int threshold) {
  int width, height, channels;
  unsigned char *h_img = stbi_load(inPath.c_str(), &width, &height, &channels, 3);
  if (!h_img) {
    std::fprintf(stderr, "Failed to load image: %s\n", inPath.c_str());
    return -1.0f;
  }
  channels = 3;  // forced 3-channel load above

  size_t rgbBytes = static_cast<size_t>(width) * height * channels;
  size_t grayBytes = static_cast<size_t>(width) * height;

  unsigned char *d_rgb, *d_gray, *d_blur, *d_edges;
  CUDA_CHECK(cudaMalloc(&d_rgb, rgbBytes));
  CUDA_CHECK(cudaMalloc(&d_gray, grayBytes));
  CUDA_CHECK(cudaMalloc(&d_blur, grayBytes));
  CUDA_CHECK(cudaMalloc(&d_edges, grayBytes));

  CUDA_CHECK(cudaMemcpy(d_rgb, h_img, rgbBytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  rgbToGrayscaleKernel<<<grid, block>>>(d_rgb, d_gray, width, height, channels);
  CUDA_CHECK(cudaGetLastError());

  gaussianBlurKernel<<<grid, block>>>(d_gray, d_blur, width, height);
  CUDA_CHECK(cudaGetLastError());

  sobelKernel<<<grid, block>>>(d_blur, d_edges, width, height, threshold);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<unsigned char> h_edges(grayBytes);
  CUDA_CHECK(cudaMemcpy(h_edges.data(), d_edges, grayBytes, cudaMemcpyDeviceToHost));

  stbi_write_png(outPath.c_str(), width, height, 1, h_edges.data(), width);

  cudaFree(d_rgb);
  cudaFree(d_gray);
  cudaFree(d_blur);
  cudaFree(d_edges);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  stbi_image_free(h_img);

  std::printf("Processed %s (%dx%d) -> %s  [%.3f ms GPU time]\n",
              inPath.c_str(), width, height, outPath.c_str(), ms);

  return ms;
}

// ---------------------------------------------------------------------------
// Directory helpers
// ---------------------------------------------------------------------------
bool hasImageExtension(const std::string &name) {
  auto endsWith = [&](const char *ext) {
    size_t len = std::strlen(ext);
    return name.size() >= len &&
           name.compare(name.size() - len, len, ext) == 0;
  };
  return endsWith(".png") || endsWith(".jpg") || endsWith(".jpeg") ||
         endsWith(".bmp") || endsWith(".PNG") || endsWith(".JPG");
}

std::vector<std::string> listImages(const std::string &dir) {
  std::vector<std::string> files;
  DIR *d = opendir(dir.c_str());
  if (!d) {
    std::fprintf(stderr, "Could not open input directory: %s\n", dir.c_str());
    return files;
  }
  struct dirent *entry;
  while ((entry = readdir(d)) != nullptr) {
    std::string name = entry->d_name;
    if (hasImageExtension(name)) {
      files.push_back(dir + "/" + name);
    }
  }
  closedir(d);
  return files;
}

std::string baseNameNoExt(const std::string &path) {
  size_t slash = path.find_last_of('/');
  std::string fname = (slash == std::string::npos) ? path : path.substr(slash + 1);
  size_t dot = fname.find_last_of('.');
  return (dot == std::string::npos) ? fname : fname.substr(0, dot);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                  "Usage: %s <input_dir> <output_dir> [--threshold N]\n",
                  argv[0]);
    return EXIT_FAILURE;
  }

  std::string inputDir = argv[1];
  std::string outputDir = argv[2];
  int threshold = 100;  // default Sobel magnitude threshold

  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
      threshold = std::atoi(argv[++i]);
    }
  }

  mkdir(outputDir.c_str(), 0755);  // no-op if it already exists

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    std::fprintf(stderr, "No CUDA-capable device found.\n");
    return EXIT_FAILURE;
  }
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("Using GPU device 0: %s\n", prop.name);

  std::vector<std::string> images = listImages(inputDir);
  if (images.empty()) {
    std::fprintf(stderr, "No images found in %s\n", inputDir.c_str());
    return EXIT_FAILURE;
  }
  std::printf("Found %zu image(s) in %s. Sobel threshold = %d\n\n",
              images.size(), inputDir.c_str(), threshold);

  std::string csvPath = outputDir + "/timing_log.csv";
  std::ofstream csv(csvPath);
  csv << "image,gpu_time_ms\n";

  float totalMs = 0.0f;
  int processed = 0;
  for (const auto &imgPath : images) {
    std::string outPath = outputDir + "/" + baseNameNoExt(imgPath) + "_edges.png";
    float ms = processImage(imgPath, outPath, threshold);
    if (ms >= 0.0f) {
      csv << baseNameNoExt(imgPath) << "," << ms << "\n";
      totalMs += ms;
      ++processed;
    }
  }
  csv.close();

  std::printf("\nDone. Processed %d/%zu images. Total GPU time = %.3f ms. "
              "Average = %.3f ms/image.\n",
              processed, images.size(), totalMs,
              processed > 0 ? totalMs / processed : 0.0f);
  std::printf("Timing log written to %s\n", csvPath.c_str());

  return EXIT_SUCCESS;
}
