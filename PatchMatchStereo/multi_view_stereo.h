#ifndef MULTI_VIEW_STEREO_H
#define MULTI_VIEW_STEREO_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <float.h>
#include <cuda_runtime_api.h> 
#include <device_launch_parameters.h> 
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "global_malloc.h"
#include "global_buffer.h"
#include "math_utils.h"
#include "display_utils.h"
#include "data_io.h"
#include "data_types.h"
#include "helper_cuda.h"


void TestHomographyWarpHost(const cv::Mat& K, const cv::Mat &R1, const cv::Mat &R2, const cv::Mat &t1, const cv::Mat &t2,
  int x, int y, float depth, float normal[3]);

void TestComputNCCHost(
  const cv::Mat &img0, const cv::Mat &img1, 
  const cv::Mat& K, 
  const cv::Mat &R1, const cv::Mat &R2, 
  const cv::Mat &t1, const cv::Mat &t2,
  int x, int y, float depth, float normal[3],
  const PatchMatchOptions *options);

#endif
