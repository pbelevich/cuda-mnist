#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <memory>
#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define FLOAT float

const size_t NUM_THREADS_X = 16;
const size_t NUM_THREADS_Y = 16;
const size_t NUM_THREADS_Z = 1;
const size_t NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y * NUM_THREADS_Z;

struct MNIST {
	MNIST(size_t m, size_t width, size_t height);
	MNIST(size_t m, size_t width, size_t height, const std::vector<FLOAT>& images, const std::vector<FLOAT>& labels);
	MNIST(std::istream &images_is, std::istream &labels_is);

	void loadImages(std::istream &images_is, size_t* m, size_t* width, size_t* height);
	void loadLabels(std::istream &labels_is, size_t* m);

	void test(size_t _m, size_t _width, size_t _height, bool output);

	size_t m, width, height;

	std::shared_ptr<FLOAT> d_x;
	std::shared_ptr<FLOAT> d_y;

	std::shared_ptr<FLOAT> h_x;
	std::shared_ptr<FLOAT> h_y;
};

class Layer {
public:
	Layer(size_t width, size_t height);
	Layer(size_t width, size_t height, const std::vector<FLOAT>& data);
	void optimize(MNIST dataset, size_t num_iterations, FLOAT learning_rate);
	std::shared_ptr<FLOAT> predict(MNIST dataset);
public:
	void forward(MNIST dataset);
	void backward(size_t m, FLOAT learning_rate);

	const size_t width, height;

	std::shared_ptr<FLOAT> d_w;
	std::shared_ptr<FLOAT> d_b;
	std::shared_ptr<FLOAT> d_dw;
	std::shared_ptr<FLOAT> d_db;

	std::shared_ptr<FLOAT> h_w;
	std::shared_ptr<FLOAT> h_b;
	std::shared_ptr<FLOAT> h_dw;
	std::shared_ptr<FLOAT> h_db;
};

using namespace std;

template<typename T>
shared_ptr<T> allocateOnDevice(size_t size) {
	T* p;
	gpuErrchk(cudaMalloc(&p, size * sizeof(T)));
	return shared_ptr<T>(p, [](T *ptr) { gpuErrchk(cudaFree(ptr)); });
}

template<typename T>
shared_ptr<T> allocateOnHost(size_t size) {
	T* p = new T[size];
	return shared_ptr<T>(p, [](T *ptr) { delete[] ptr; });
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

std::istream& read_int32(std::istream& is, int32_t &x) {
	unsigned char buffer[4];
	auto &res = is.read(reinterpret_cast<char *>(buffer), sizeof buffer);
	x = (int)buffer[3] | (int)buffer[2] << 8 | (int)buffer[1] << 16 | (int)buffer[0] << 24;
	return res;
}

MNIST::MNIST(std::istream &images_is, std::istream &labels_is) {
	size_t images_m, images_width, images_height;
	loadImages(images_is, &images_m, &images_width, &images_height);
	size_t labels_m;
	loadLabels(labels_is, &labels_m);
	assert(images_m == labels_m);
	m = images_m;
	width = images_width;
	height = images_height;
}

void MNIST::loadImages(std::istream &is, size_t* m, size_t* width, size_t* height) {
	int32_t magic, number_of_images, number_of_rows, number_of_columns;
	read_int32(is, magic);
	read_int32(is, number_of_images);
	read_int32(is, number_of_rows);
	read_int32(is, number_of_columns);

	*m = number_of_images;
	*width = number_of_columns;
	*height = number_of_rows;
	size_t size = (*m) * (*width) * (*height);

	h_x = allocateOnHost<FLOAT>(size);
	auto chars = allocateOnHost<unsigned char>(size);
	is.read(reinterpret_cast<char*>(chars.get()), size);
	for (int32_t i = 0; i < size; i++) {
		h_x.get()[i] = static_cast<FLOAT>((chars.get()[i]) / 255.0f);
	}

	d_x = allocateOnDevice<FLOAT>(size);
	gpuErrchk(cudaMemcpy(d_x.get(), h_x.get(), size * sizeof(FLOAT), cudaMemcpyHostToDevice));
}

void MNIST::loadLabels(std::istream &is, size_t* m) {
	int32_t magic, number_of_labels;
	read_int32(is, magic);
	read_int32(is, number_of_labels);

	*m = number_of_labels;
	size_t size = (*m);

	h_y = allocateOnHost<FLOAT>(size);
	auto chars = allocateOnHost<unsigned char>(size);
	is.read(reinterpret_cast<char*>(chars.get()), size);
	for (int32_t i = 0; i < size; i++) {
		h_y.get()[i] = static_cast<FLOAT>(chars.get()[i] == 5 ? 1.0f : 0.0f);
	}

	d_y = allocateOnDevice<FLOAT>(size);
	gpuErrchk(cudaMemcpy(d_y.get(), h_y.get(), size * sizeof(FLOAT), cudaMemcpyHostToDevice));
}

void MNIST::test(size_t _m, size_t _width, size_t _height, bool output) {
	assert(m == _m);
	assert(width == _width);
	assert(height == _height);
	for (size_t i = 0; i < 20; i++) {
		if (output) {
			for (size_t r = 0; r < 28; r++) {
				for (size_t c = 0; c < 28; c++) {
					char ch = (char)(h_x.get()[i * 28 * 28 + r * 28 + c] * 255);
					cout << ch << " ";
				}
				cout << endl;
			}
		}
		//assert(h_y.get()[i] == (i == 0 || i == 11 ? 1.0f : 0.0f));
	}
}

Layer::Layer(size_t width, size_t height) : 
	width(width), height(height),
	h_w(allocateOnHost<FLOAT>(width * height)),
	h_b(allocateOnHost<FLOAT>(1)),
	h_dw(allocateOnHost<FLOAT>(width * height)),
	h_db(allocateOnHost<FLOAT>(1)),
	d_w(allocateOnDevice<FLOAT>(width * height)),
	d_b(allocateOnDevice<FLOAT>(1)),
	d_dw(allocateOnDevice<FLOAT>(width * height)),
	d_db(allocateOnDevice<FLOAT>(1)) {

	memsetOnHost(h_w.get(), 0.0f, width * height);
	memsetOnHost(h_b.get(), 0.0f, 1);
	memsetOnHost(h_dw.get(), 0.0f, width * height);
	memsetOnHost(h_db.get(), 0.0f, 1);

	gpuErrchk(cudaMemcpy(d_w.get(), h_w.get(), width * height * sizeof(FLOAT), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_b.get(), h_b.get(), sizeof(FLOAT), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_dw.get(), h_dw.get(), width * height * sizeof(FLOAT), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_db.get(), h_db.get(), sizeof(FLOAT), cudaMemcpyHostToDevice));
}

__device__ __host__ FLOAT sigmoid(FLOAT x) {
	return 1.0f / (1 + expf(-x));
}

// blocks(1, 1, m)
// threads(NUM_THREADS_X, NUM_THREADS_X, 1)
__global__ void product(FLOAT *x, FLOAT *w, FLOAT* b, FLOAT* y, size_t width, size_t height, size_t m) {
	__shared__ FLOAT cache[NUM_THREADS];

	const size_t i = threadIdx.x;
	const size_t j = threadIdx.y;
	const size_t k = blockIdx.z;

	const size_t cid = (j * NUM_THREADS_X) + i;

	FLOAT temp = 0.0f;
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
		const FLOAT a = sigmoid(cache[0] + *b);
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
__global__ void cost(FLOAT *y, FLOAT *a, FLOAT* cost_partial, size_t m) {
	__shared__ FLOAT cache[NUM_THREADS];

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
template<typename T>
__global__ void memsetOnDevice(T* ts, T val, size_t size) {
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size) {
		ts[i] = val;
	}
}

// blocks(width/NUM_THREADS_X, height/NUM_THREADS_Y, m)
// threads(NUM_THREADS_X, NUM_THREADS_Y, 1)
__global__ void weights_gradient(FLOAT *x, FLOAT *y, FLOAT *a, FLOAT *dw, size_t width, size_t height) {
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
__global__ void bias_gradient(FLOAT *y, FLOAT *a, FLOAT *db_partial, size_t m) {
	__shared__ FLOAT cache[NUM_THREADS];

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
	auto d_a = allocateOnDevice<FLOAT>(m);
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
	auto d_cost_partial = allocateOnDevice<FLOAT>(blocks.x);
	cost<<<blocks, threads>>>(d_y.get(), d_a.get(), d_cost_partial.get(), m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO: sum up d_cost_partial and divide by m
	const auto h_cost = sumOnHost<FLOAT>(d_cost_partial.get(), blocks.x) / m;

	//const auto h_cost_expected = costOnHost(h_y.get(), h_a_actual.get(), m);

	//const auto cost_diff = abs(h_cost - h_cost_expected);
	//assert(cost_diff < 1e-3);
	cout << h_cost << endl;

	threads = NUM_THREADS;
	blocks = (width * height + threads.x - 1) / threads.x;
	memsetOnDevice<<<blocks, threads>>>(d_dw.get(), 0.0f, width * height);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	threads = {(unsigned int)NUM_THREADS_X, (unsigned int)NUM_THREADS_Y, 1};
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
	auto d_db_partial = allocateOnDevice<FLOAT>(blocks.x);
	bias_gradient<<<blocks, threads>>>(d_y.get(), d_a.get(), d_db_partial.get(), m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	// TODO: sum up d_db_partial and divide by m
	auto h_db = sumOnHost<FLOAT>(d_db_partial.get(), blocks.x) / m;
	gpuErrchk(cudaMemcpy(d_db.get(), &h_db, sizeof(FLOAT), cudaMemcpyHostToDevice));

	//auto h_db_expected = bias_gradientOnHost(h_y.get(), h_a_actual.get(), m);
	//const auto h_db_diff = abs(h_db - h_db_expected);
	//assert(h_db_diff < 1e-1);
}

// blocks(width/NUM_THREADS_X, height/NUM_THREADS_Y, m)
// threads(NUM_THREADS_X, NUM_THREADS_Y, 1)
__global__ void weights_backward(FLOAT *w, FLOAT *dw, size_t width, size_t height, size_t m, FLOAT learning_rate) {
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t wid = (j * width) + i;

	if (i < width && j < height) {
		w[wid] -= learning_rate * dw[wid] / m;
	}
}

void weights_backwardOnHost(FLOAT *w, FLOAT *dw, size_t width, size_t height, size_t m, FLOAT learning_rate) {
	for (size_t j = 0; j < height; j++) {
		for (size_t i = 0; i < width; i++) {
			const size_t wid = (j * width) + i;
			w[wid] -= learning_rate * dw[wid] / m;
		}
	}
}

// blocks(1)
// threads(1)
__global__ void bias_backward(FLOAT *b, FLOAT *db, FLOAT learning_rate) {
	b[0] -= learning_rate * db[0];
}

void Layer::backward(size_t m, FLOAT learning_rate) {
	dim3 threads(NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, 1);
	weights_backward<<<blocks, threads>>>(d_w.get(), d_dw.get(), width, height, m, learning_rate);

	weights_backwardOnHost(h_w.get(), h_dw.get(), width, height, m, learning_rate);

	threads = 1;
	blocks = 1;
	bias_backward<<<blocks, threads>>>(d_b.get(), d_db.get(), learning_rate);
}

void Layer::optimize(MNIST dataset, size_t num_iterations, FLOAT learning_rate) {
	for (size_t i = 0; i < num_iterations; ++i) {
		forward(dataset);
		backward(dataset.m, learning_rate);
	}
}

std::shared_ptr<FLOAT> Layer::predict(MNIST dataset) {
	std::shared_ptr<FLOAT> d_y_pred = allocateOnDevice<FLOAT>(dataset.m);

	dim3 threads(NUM_THREADS_X, NUM_THREADS_Y, 1);
	dim3 blocks(1, 1, dataset.m);
	product<<<blocks, threads>>>(dataset.d_x.get(), d_w.get(), d_b.get(), d_y_pred.get(), dataset.width, dataset.height, dataset.m);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	return d_y_pred;
}

int main() {
	ifstream train_images_is("..\\train-images.idx3-ubyte", std::ios::in | std::ios::binary);
	ifstream train_labels_is("..\\train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	MNIST train_dataset(train_images_is, train_labels_is);
	train_dataset.test(60000, 28, 28, false);

	Layer l(28, 28);
	l.optimize(train_dataset, 1000, 1e-3f);

	ifstream test_images_is("..\\t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
	ifstream test_labels_is("..\\t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
	MNIST test_dataset(test_images_is, test_labels_is);
	test_dataset.test(10000, 28, 28, false);

	auto d_y_pred = l.predict(test_dataset);
	auto h_y_pred = allocateOnHost<FLOAT>(test_dataset.m);
	gpuErrchk(cudaMemcpy(h_y_pred.get(), d_y_pred.get(), test_dataset.m * sizeof(FLOAT), cudaMemcpyDeviceToHost));

	for (size_t k = 0; k < test_dataset.m; k++) {
		cout << test_dataset.h_y.get()[k] << " " << h_y_pred.get()[k] << endl;
	}

	return 0;
}
