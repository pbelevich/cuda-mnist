#include "layer.h"
#include "util.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <iostream>
#include <cassert>

using namespace std;

const size_t NUM_THREADS_X = 8;
const size_t NUM_THREADS_Y = 8;
const size_t NUM_THREADS_Z = 1;
const size_t NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y * NUM_THREADS_Z;

Layer::Layer(size_t width, size_t height) :
	width(width), height(height),
	h_w(allocateOnHost<float>(width * height)),
	h_b(allocateOnHost<float>(1)),
	h_dw(allocateOnHost<float>(width * height)),
	h_db(allocateOnHost<float>(1)),
	d_w(allocateOnDevice<float>(width * height)),
	d_b(allocateOnDevice<float>(1)),
	d_dw(allocateOnDevice<float>(width * height)),
	d_db(allocateOnDevice<float>(1)) {

	memsetOnHost(h_w.get(), 0.0f, width * height);
	memsetOnHost(h_b.get(), 0.0f, 1);
	memsetOnHost(h_dw.get(), 0.0f, width * height);
	memsetOnHost(h_db.get(), 0.0f, 1);

	gpuErrchk(cudaMemcpy(d_w.get(), h_w.get(), width * height * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b.get(), h_b.get(), sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_dw.get(), h_dw.get(), width * height * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_db.get(), h_db.get(), sizeof(float), cudaMemcpyHostToDevice));
}

__device__ __host__ float sigmoid(float x) {
	return 1.0f / (1 + expf(-x));
}

// blocks(1, 1, m)
// threads(NUM_THREADS_X, NUM_THREADS_X, 1)
__global__ void product(float *x, float *w, float* b, float* y, size_t width, size_t height, size_t m) {
	__shared__ float cache[NUM_THREADS];

	const size_t i = threadIdx.x;
	const size_t j = threadIdx.y;
	const size_t k = blockIdx.z;

	const size_t cid = (j * NUM_THREADS_X) + i;

	float temp = 0.0f;
	for (auto ii = i; ii < width; ii += blockDim.x) {
		for (auto jj = j; jj < height; jj += blockDim.y) {
			const size_t wid = (jj * width) + ii;
			const size_t xid = (k * width * height) + wid;
			temp += w[wid] * x[xid];
		}
	}
	cache[cid] = temp;
	__syncthreads();

	const size_t size = NUM_THREADS;
	for (size_t half = size / 2; half > 0; half /= 2) {
		if (cid < half) {
			cache[cid] += cache[cid + half];
		}
		__syncthreads();
	}

	if (cid == 0) {
		const float a = sigmoid(cache[0] + *b);
		assert(a != 0.0f && a != 1.0f);
		y[k] = a;
	}
}

//void productOnHost(FLOAT *x, FLOAT *w, FLOAT* b, FLOAT* y, size_t width, size_t height, size_t m) {
//	for (size_t k = 0; k < m; k++) {
//		y[k] = *b;
//		for (size_t j = 0; j < height; j++) {
//			for (size_t i = 0; i < width; i++) {
//				const size_t wid = (j * width) + i;
//				const size_t xid = (k * width * height) + wid;
//				y[k] += w[width * i + j] * x[xid];
//			}
//		}
//		y[k] = sigmoid(y[k]);
//	}
//}

// blocks(m/NUM_THREADS, 1, 1)
// threads(NUM_THREADS, 1, 1)
__global__ void cost(float *y, float *a, float* cost_partial, size_t m) {
	__shared__ float cache[NUM_THREADS];

	const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t cid = threadIdx.x;
	const size_t bid = blockIdx.x;

	float temp = 0.0f;
	if (tid < m) {
		temp = y[tid] * logf(a[tid]) + (1 - y[tid]) * (logf(1 - a[tid]));
	}
	cache[cid] = temp;
	__syncthreads();

	const size_t size = NUM_THREADS;
	for (size_t half = size / 2; half > 0; half /= 2) {
		if (cid < half) {
			cache[cid] += cache[cid + half];
		}
		__syncthreads();
	}

	if (cid == 0) {
		cost_partial[bid] = -cache[0];
	}
}

//FLOAT costOnHost(FLOAT *y, FLOAT *a, size_t m) {
//	FLOAT cost = 0.0f;
//	for (size_t tid = 0; tid < m; tid++) {
//		cost += y[tid] * logf(a[tid]) + (1 - y[tid]) * (logf(1 - a[tid]));
//	}
//	return -cost / m;
//}

// blocks(size/NUM_THREADS)
// threads(NUM_THREADS)
__global__ void memsetOnDevice(float* ts, float val, size_t size) {
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		ts[i] = val;
	}
}


// blocks(width/NUM_THREADS_X, height/NUM_THREADS_Y, m)
// threads(NUM_THREADS_X, NUM_THREADS_Y, 1)
__global__ void weights_gradient(float *x, float *y, float *a, float *dw, size_t width, size_t height) {
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockIdx.z;

	const size_t wid = (j * width) + i;
	const size_t xid = (k * width * height) + wid;

	//printf("wid1 = %d xid1 = %d %f %f %f\n", wid, xid, x[xid], a[k], y[k]);

	atomicAdd(&dw[wid], x[xid] * (a[k] - y[k]));
}

//void weights_gradientOnHost(FLOAT *x, FLOAT *y, FLOAT *a, FLOAT *dw, size_t width, size_t height, size_t m) {
//	for (size_t j = 0; j < height; j++) {
//		for (size_t i = 0; i < width; i++) {
//			const size_t wid = (j * width) + i;
//			dw[wid] = 0.0f;
//			for (size_t k = 0; k < m; k++) {
//				const size_t xid = (k * width * height) + wid;
//				dw[wid] += x[xid] * (a[k] - y[k]);
//				//printf("wid2 = %d xid2 = %d %f %f %f\n", wid, xid, x[xid], a[k], y[k]);
//			}
//		}
//	}
//}

// blocks(m/NUM_THREADS, 1, 1)
// threads(NUM_THREADS, 1, 1)
__global__ void bias_gradient(float *y, float *a, float *db_partial, size_t m) {
	__shared__ float cache[NUM_THREADS];

	const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t cid = threadIdx.x;
	const size_t bid = blockIdx.x;

	float temp = 0.0f;
	if (tid < m) {
		temp = a[tid] - y[tid];
	}
	cache[cid] = temp;
	__syncthreads();

	const size_t size = NUM_THREADS;
	for (size_t half = size / 2; half > 0; half /= 2) {
		if (cid < half) {
			cache[cid] += cache[cid + half];
		}
		__syncthreads();
	}

	if (cid == 0) {
		db_partial[bid] = cache[0];
	}
}

//FLOAT bias_gradientOnHost(FLOAT *y, FLOAT *a, size_t m) {
//	FLOAT bias = 0.0f;
//	for (size_t tid = 0; tid < m; tid++) {
//		bias += a[tid] - y[tid];
//	}
//	return bias;
//}

void Layer::forward(MNIST dataset) {
	size_t width = dataset.width;
	size_t height = dataset.height;
	size_t m = dataset.m;

	const auto& d_x = dataset.d_x;
	const auto& d_y = dataset.d_y;
	const auto& h_x = dataset.h_x;
	const auto& h_y = dataset.h_y;

	dim3 threads(NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 blocks(1, 1, m);
	auto d_a = allocateOnDevice<float>(m);
	product<<<blocks, threads>>>(d_x.get(), d_w.get(), d_b.get(), d_a.get(), width, height, m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//auto h_a_actual = allocateOnHost<FLOAT>(m);
	//gpuErrchk(cudaMemcpy(h_a_actual.get(), d_a.get(), m*sizeof(FLOAT), cudaMemcpyDeviceToHost));

	//auto h_a_expected = allocateOnHost<FLOAT>(m);
	//productOnHost(h_x.get(), h_w.get(), h_b.get(), h_a_expected.get(), width, height, m);

	//assertEqual(h_a_actual.get(), h_a_expected.get(), m);

	threads = NUM_THREADS;
	blocks = (m + threads.x - 1) / threads.x;
	auto d_cost_partial = allocateOnDevice<float>(blocks.x);
	cost<<<blocks, threads>>>(d_y.get(), d_a.get(), d_cost_partial.get(), m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO: sum up d_cost_partial and divide by m
	const auto h_cost = sumOnHost<float>(d_cost_partial.get(), blocks.x) / m;

	//const auto h_cost_expected = costOnHost(h_y.get(), h_a_actual.get(), m);

	//const auto cost_diff = abs(h_cost - h_cost_expected);
	//assert(cost_diff < 1e-3);
	cout << h_cost << endl;

	threads = NUM_THREADS;
	blocks = (width * height + threads.x - 1) / threads.x;
	memsetOnDevice<<<blocks, threads>>>(d_dw.get(), 0.0f, width * height);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	threads = { (unsigned int)NUM_THREADS_X, (unsigned int)NUM_THREADS_Y, 1 };
	blocks = {
		(unsigned int)((width + threads.x - 1) / threads.x),
		(unsigned int)((height + threads.y - 1) / threads.y),
		(unsigned int)m
	};
	weights_gradient<<<blocks, threads>>>(d_x.get(), d_y.get(), d_a.get(), d_dw.get(), width, height);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO: divide d_dw by m

	//gpuErrchk(cudaMemcpy(h_dw.get(), d_dw.get(), width * height * sizeof(FLOAT), cudaMemcpyDeviceToHost));

	//auto h_dw_expected = allocateOnHost<FLOAT>(width * height);
	//weights_gradientOnHost(h_x.get(), h_y.get(), h_a_actual.get(), h_dw_expected.get(), width, height, m);

	//assertEqual(h_dw.get(), h_dw_expected.get(), width * height);

	threads = NUM_THREADS;
	blocks = (m + threads.x - 1) / threads.x;
	auto d_db_partial = allocateOnDevice<float>(blocks.x);
	bias_gradient<<<blocks, threads>>>(d_y.get(), d_a.get(), d_db_partial.get(), m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO: sum up d_db_partial and divide by m
	auto h_db = sumOnHost<float>(d_db_partial.get(), blocks.x) / m;
	gpuErrchk(cudaMemcpy(d_db.get(), &h_db, sizeof(float), cudaMemcpyHostToDevice));

	//auto h_db_expected = bias_gradientOnHost(h_y.get(), h_a_actual.get(), m);
	//const auto h_db_diff = abs(h_db - h_db_expected);
	//assert(h_db_diff < 1e-1);
}

// blocks(width/NUM_THREADS_X, height/NUM_THREADS_Y, m)
// threads(NUM_THREADS_X, NUM_THREADS_Y, 1)
__global__ void weights_backward(float *w, float *dw, size_t width, size_t height, size_t m, float learning_rate) {
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t wid = (j * width) + i;

	if (i < width && j < height) {
		w[wid] -= learning_rate * dw[wid] / m;
	}
}

void weights_backwardOnHost(float *w, float *dw, size_t width, size_t height, size_t m, float learning_rate) {
	for (size_t j = 0; j < height; j++) {
		for (size_t i = 0; i < width; i++) {
			const size_t wid = (j * width) + i;
			w[wid] -= learning_rate * dw[wid] / m;
		}
	}
}

// blocks(1)
// threads(1)
__global__ void bias_backward(float *b, float *db, float learning_rate) {
	b[0] -= learning_rate * db[0];
}

void Layer::backward(size_t m, float learning_rate) {
	dim3 threads(NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
	weights_backward<<<blocks, threads>>>(d_w.get(), d_dw.get(), width, height, m, learning_rate);

	weights_backwardOnHost(h_w.get(), h_dw.get(), width, height, m, learning_rate);

	threads = 1;
	blocks = 1;
	bias_backward<<<blocks, threads>>>(d_b.get(), d_db.get(), learning_rate);
}

void Layer::optimize(MNIST dataset, size_t num_iterations, float learning_rate) {
	for (size_t i = 0; i < num_iterations; ++i) {
		forward(dataset);
		backward(dataset.m, learning_rate);
	}
}

std::shared_ptr<float> Layer::predict(MNIST dataset) {
	std::shared_ptr<float> d_y_pred = allocateOnDevice<float>(dataset.m);

	dim3 threads(NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 blocks(1, 1, dataset.m);
	product<<<blocks, threads>>>(dataset.d_x.get(), d_w.get(), d_b.get(), d_y_pred.get(), dataset.width, dataset.height, dataset.m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return d_y_pred;
}
