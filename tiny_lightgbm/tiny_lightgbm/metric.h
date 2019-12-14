#pragma once

#include "dataset.h"


namespace Tiny_LightGBM {


class Metric {


public:
	static Metric* CreateMetric(int type);

	virtual void Init(const Metadata& metadata, int num_data) = 0;

};





}