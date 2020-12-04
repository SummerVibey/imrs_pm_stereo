#ifndef GLOBAL_BUFFER_H
#define GLOBAL_BUFFER_H

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
class GlobalBuffer 
{
private:
	T *m_data;
	size_t m_size;

public:
	GlobalBuffer()
		: m_data(nullptr)
		, m_size(0)
	{ }

	explicit GlobalBuffer(size_t n)
		: m_data(nullptr)
		, m_size(0)
	{
		allocate(n);
	}

	GlobalBuffer(const GlobalBuffer&) = delete;

	GlobalBuffer(GlobalBuffer&& obj)
		: m_data(obj.m_data)
		, m_size(obj.m_size)
	{
		obj.m_data = nullptr;
		obj.m_size = 0;
	}

	~GlobalBuffer(){
		destroy();
	}


	void allocate(size_t n){
		if(m_data && m_size >= n)
			return;
		destroy();
		assert(cudaSuccess == cudaMallocManaged(reinterpret_cast<void **>(&m_data), sizeof(T) * n));
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

	GlobalBuffer& operator=(const GlobalBuffer&) = delete;

	GlobalBuffer& operator=(GlobalBuffer&& obj){
		m_data = obj.m_data;
		m_size = obj.m_size;
		obj.m_data = nullptr;
		obj.m_size = 0;
		return *this;
	}

	int size() const {
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

