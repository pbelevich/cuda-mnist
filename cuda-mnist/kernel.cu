#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "mnist.h"
#include "layer.h"
#include "util.h"

#include <iostream>
#include <fstream>

using namespace std;

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
	auto h_y_pred = allocateOnHost<float>(test_dataset.m);
	gpuErrchk(cudaMemcpy(h_y_pred.get(), d_y_pred.get(), test_dataset.m * sizeof(float), cudaMemcpyDeviceToHost));

	for (size_t k = 0; k < test_dataset.m; k++) {
		cout << test_dataset.h_y.get()[k] << " " << h_y_pred.get()[k] << endl;
	}

	return 0;
}
