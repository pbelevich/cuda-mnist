#pragma once

#include "mnist.h"

class Layer {
public:
	Layer(size_t width, size_t height);
	void optimize(MNIST dataset, size_t num_iterations, float learning_rate);
	std::shared_ptr<float> predict(MNIST dataset);
public:
	void forward(MNIST dataset);
	void backward(size_t m, float learning_rate);

	const size_t width, height;

	std::shared_ptr<float> d_w;
	std::shared_ptr<float> d_b;
	std::shared_ptr<float> d_dw;
	std::shared_ptr<float> d_db;

	std::shared_ptr<float> h_w;
	std::shared_ptr<float> h_b;
	std::shared_ptr<float> h_dw;
	std::shared_ptr<float> h_db;
};
