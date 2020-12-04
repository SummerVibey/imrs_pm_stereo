#ifndef GLOBAL_MALLOC_H
#define GLOBAL_MALLOC_H

#include <iostream>

// Includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "helper_cuda.h"



class GlobalMalloc 
{
public:
  void *operator new(size_t len) {
    void *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaFree(ptr);
  }
};


#endif