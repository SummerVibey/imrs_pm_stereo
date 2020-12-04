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
	DeviceBuffer()
		: m_data(nullptr)
		, m_size(0)
	{ }

	explicit DeviceBuffer(size_t n)
		: m_data(nullptr)
		, m_size(0)
	{
		allocate(n);
	}

	DeviceBuffer(const DeviceBuffer&) = delete;

	DeviceBuffer(DeviceBuffer&& obj)
		: m_data(obj.m_data)
		, m_size(obj.m_size)
	{
		obj.m_data = nullptr;
		obj.m_size = 0;
	}

	~DeviceBuffer(){
		destroy();
	}


	void allocate(size_t n){
		if(m_data && m_size >= n)
			return;
		destroy();
		assert(cudaSuccess == cudaMalloc(reinterpret_cast<void **>(&m_data), sizeof(T) * n));
		m_size = n;
	}

	void destroy(){
		if(m_data)
			assert(cudaSuccess == cudaFree(m_data));

		m_data = nullptr;
		m_size = 0;
	}

	void fillZero(){
		assert(cudaSuccess == cudaMemset(m_data, 0, sizeof(T) * m_size));
	}

	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	DeviceBuffer& operator=(DeviceBuffer&& obj){
		m_data = obj.m_data;
		m_size = obj.m_size;
		obj.m_data = nullptr;
		obj.m_size = 0;
		return *this;
	}


	size_t size() const {
		return m_size;
	}

	const T *data() const {
		return m_data;
	}

	T *data(){
		return m_data;
	}

};

#endif
