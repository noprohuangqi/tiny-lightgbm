

#include "dataset.h"



namespace Tiny_LightGBM {


void Metadata::SetLabel(const float* label , int len) {

	label_ = std::vector<float>(len);

	for (int i = 0; i < len; ++i) {
		label_[i] = label[i];
	}

}

}