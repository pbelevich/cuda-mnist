#include "mnist.h"
#include "util.h"
#include <cassert>
#include <iostream>

using namespace std;

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

	h_x = allocateOnHost<float>(size);
	auto chars = allocateOnHost<unsigned char>(size);
	is.read(reinterpret_cast<char*>(chars.get()), size);
	for (int32_t i = 0; i < size; i++) {
		h_x.get()[i] = static_cast<float>((chars.get()[i]) / 255.0f);
	}

	d_x = allocateOnDevice<float>(size);
	gpuErrchk(cudaMemcpy(d_x.get(), h_x.get(), size * sizeof(float), cudaMemcpyHostToDevice));
}

void MNIST::loadLabels(std::istream &is, size_t* m) {
	int32_t magic, number_of_labels;
	read_int32(is, magic);
	read_int32(is, number_of_labels);

	*m = number_of_labels;
	size_t size = (*m);

	h_y = allocateOnHost<float>(size);
	auto chars = allocateOnHost<unsigned char>(size);
	is.read(reinterpret_cast<char*>(chars.get()), size);
	for (int32_t i = 0; i < size; i++) {
		h_y.get()[i] = static_cast<float>(chars.get()[i] == 5 ? 1.0f : 0.0f);
	}

	d_y = allocateOnDevice<float>(size);
	gpuErrchk(cudaMemcpy(d_y.get(), h_y.get(), size * sizeof(float), cudaMemcpyHostToDevice));
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
