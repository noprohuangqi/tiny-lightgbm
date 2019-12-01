#pragma once

#include "define.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <iterator>
#include <type_traits>
#include <iomanip>

namespace Tiny_LightGBM {

namespace Utils {

template <typename T>
inline static std::vector<T*> Vector2Ptr(std::vector<std::vector<T>>& data) {
	std::vector<T*> ptr(data.size());

	for (int i = 0; i < data.size(); ++i) {
		ptr[i] = data[i].data();
	}
	return ptr;

}


template <typename T>
inline static std::vector<int> VectorSize(const std::vector<std::vector<T>>& data) {

	std::vector<int> ret(data.size());

	for (int i = 0; i < data.size(); ++i) {
		ret[i] = static_cast<int>(data[i].size());
	}
	return ret;
}


inline static bool CheckDoubleEqualOrdered(double a, double b) {

	double upper = std::nextafter(a, INFINITY);

	return b <= upper;

}

inline static double GetDoubleUpperBound(double a) {
	return std::nextafter(a, INFINITY);	
}

template <typename T>
static int Sign(T x) {
	return (x > T(0)) - (x < T(0));
}

template<typename T>
inline std::vector<const T*> ConstPtrInVectorWrapper(std::vector<std::unique_ptr<T>>& input) {
	
	std::vector<const T*> ret;
	for (size_t i = 0; i < input.size(); ++i) {

		ret.push_back(input.at(i).get());
	}
	return ret;
}

}
}

