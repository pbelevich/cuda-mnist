#include "util.h"

std::istream& read_int32(std::istream& is, int32_t &x) {
	unsigned char buffer[4];
	auto &res = is.read(reinterpret_cast<char *>(buffer), sizeof buffer);
	x = (int)buffer[3] | (int)buffer[2] << 8 | (int)buffer[1] << 16 | (int)buffer[0] << 24;
	return res;
}
