#pragma once



#include "metric.h"

namespace Tiny_LightGBM {


template<typename PointWiseLossCalculator>
class RegressionMetric :public Metric {
public:

	RegressionMetric() {}

	void Init(const Metadata& metadata, int num_data) {
		num_data_ = num_data;
		label_ = metadata.label();

	}
private:

	int num_data_;

	const float* label_;




};

class L2Metric :public RegressionMetric<L2Metric> {
public:
	L2Metric():RegressionMetric<L2Metric>() {}
private:


};



}

