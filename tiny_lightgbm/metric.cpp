

#include "metric.h"
#include "regression_metric.hpp"

namespace Tiny_LightGBM {

Metric* Metric::CreateMetric(int type) {

	if (type == 1) {
		return new L2Metric();
	}
}




}