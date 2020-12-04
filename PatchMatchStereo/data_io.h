#ifndef DATA_IO_H
#define DATA_IO_H

#include "data_types.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>


void ConvertPlaneToDispAndNormal(DispPlane *plane, cv::Mat_<float>& disp, cv::Mat_<cv::Vec3f>& norm, int width, int height);


#endif