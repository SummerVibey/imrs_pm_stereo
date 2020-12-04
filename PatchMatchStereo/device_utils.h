#ifndef DEVICE_UTILS_H
#define DEVICE_UTILS_H



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

#include <vector>
#include <string>
#include <iostream>

#include "helper_cuda.h"

static void SelectCudaDevice()
{
  int deviceCount = 0;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "There is no cuda capable device!\n");
    exit(EXIT_FAILURE);
  } 
  std::cout << "Detected " << deviceCount << " devices!" << std::endl;
  std::vector<int> usableDevices;
  std::vector<std::string> usableDeviceNames;
  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      if (prop.major >= 3 && prop.minor >= 0) {
        usableDevices.push_back(i);
        usableDeviceNames.push_back(std::string(prop.name));
      } else {
        std::cout << "CUDA capable device " << std::string(prop.name)
              << " is only compute cabability " << prop.major << '.'
              << prop.minor << std::endl;
      }
    } else {
      std::cout << "Could not check device properties for one of the cuda "
              "devices!" << std::endl;
    }
  }
  if(usableDevices.empty()) {
    fprintf(stderr, "There is no cuda device supporting RumMultiViewStereo!\n");
    exit(EXIT_FAILURE);
  }
  std::cout << "Detected compatible device: " << usableDeviceNames[0] << std::endl;
  checkCudaErrors(cudaSetDevice(usableDevices[0]));
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*128);
}


#endif