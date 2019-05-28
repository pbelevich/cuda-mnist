#pragma once

#include <memory>
#include <istream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template<typename T>
std::shared_ptr<T> allocateOnDevice(size_t size) {
	T* p;
	gpuErrchk(cudaMalloc(&p, size * sizeof(T)));
	return std::shared_ptr<T>(p, [](T *ptr) { gpuErrchk(cudaFree(ptr)); });
}

template<typename T>
std::shared_ptr<T> allocateOnHost(size_t size) {
	T* p = new T[size];
	return std::shared_ptr<T>(p, [](T *ptr) { delete[] ptr; });
}

template<typename T>
T sumOnHost(T* data, size_t size) {
	auto s_p = allocateOnHost<T>(size);
	gpuErrchk(cudaMemcpy(s_p.get(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
	T sum = T();
	for (size_t i = 0; i < size; ++i) {
		sum += s_p.get()[i];
	}
	return sum;
}

//template<typename T>
//void assertEqual(T* actual, T* expected, size_t size) {
//	for (size_t i = 0; i < size; i++) {
//		auto diff = abs(actual[i] - expected[i]);
//		if (diff >= 1e-1) {
//			cout << "";
//		}
//		assert(diff < 1e-1);
//	}
//}

template<typename T>
void memsetOnHost(T* ts, T val, size_t size) {
	for (size_t i = 0; i < size; i++) {
		ts[i] = val;
	}
}

std::istream& read_int32(std::istream& is, int32_t &x);
