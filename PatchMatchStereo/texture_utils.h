#ifndef TEXTURE_UTILS_H
#define TEXTURE_UTILS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>

#include <opencv2/core/core.hpp>

#include "global_malloc.h"

void CreateTextureObject(const cv::Mat &img, cudaTextureObject_t& tex, cudaArray *arr);

void DestroyTextureObject(cudaTextureObject_t& tex, cudaArray *arr);

#endif