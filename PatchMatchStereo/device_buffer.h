#ifndef DEVICE_BUFFER_HPP
#define DEVICE_BUFFER_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <assert.h>

#include "helper_cuda.h"


template <typename T>
class DeviceBuffer 
{
private:
	T *m_data;
	size_t m_size;

public:
	__device__ DeviceBuffer()
		: m_data(nullptr)
		, m_size(0)
	{ }

	__device__ explicit DeviceBuffer(size_t n)
		: m_data(nullptr)
		, m_size(0)
	{
		allocate(n);
	}

	__device__ DeviceBuffer(const DeviceBuffer&) = delete;

	__device__ DeviceBuffer(DeviceBuffer&& obj)
		: m_data(obj.m_data)
		, m_size(obj.m_size)
	{
		obj.m_data = nullptr;
		obj.m_size = 0;
	}

	__device__ ~DeviceBuffer(){
		destroy();
	}

  __device__ T* ptr(size_t n) const {
    return &m_data[n];
  }

  __device__ T& at(size_t n) {
    return m_data[n];
  }


	__device__ void allocate(size_t n){
		if(m_data && m_size >= n)
			return;
		destroy();
		assert(cudaSuccess == cudaMalloc(reinterpret_cast<void **>(&m_data), sizeof(T) * n));
		m_size = n;
	}

	__device__ void destroy(){
		if(m_data)
			assert(cudaSuccess == cudaFree(m_data));

		m_data = nullptr;
		m_size = 0;
	}

	__device__ void fillZero(){
		assert(cudaSuccess == cudaMemset(m_data, 0, sizeof(T) * m_size));
	}

	__device__ DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	__device__ DeviceBuffer& operator=(DeviceBuffer&& obj){
		m_data = obj.m_data;
		m_size = obj.m_size;
		obj.m_data = nullptr;
		obj.m_size = 0;
		return *this;
	}


	__device__ size_t size() const {
		return m_size;
	}

	__device__ const T *data() const {
		return m_data;
	}

	__device__ T *data(){
		return m_data;
	}

};

#endif
