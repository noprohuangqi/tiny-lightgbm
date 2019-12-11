#pragma once


namespace Tiny_LightGBM {


struct SplitInfo {
public:
	/*! \brief Feature index */
	int feature = -1;
	/*! \brief Split threshold */
	uint32_t threshold = 0;
	/*! \brief Left number of data after split */
	int left_count = 0;
	/*! \brief Right number of data after split */
	int right_count = 0;
	int num_cat_threshold = 0;
	/*! \brief Left output after split */
	double left_output = 0.0;
	/*! \brief Right output after split */
	double right_output = 0.0;
	/*! \brief Split gain */
	double gain = -std::numeric_limits<float>::infinity();
	/*! \brief Left sum gradient after split */
	double left_sum_gradient = 0;
	/*! \brief Left sum hessian after split */
	double left_sum_hessian = 0;
	/*! \brief Right sum gradient after split */
	double right_sum_gradient = 0;
	/*! \brief Right sum hessian after split */
	double right_sum_hessian = 0;
	std::vector<uint32_t> cat_threshold;
	/*! \brief True if default split is left */
	bool default_left = true;
	int monotone_type = 0;
	double min_constraint = -std::numeric_limits<double>::max();
	double max_constraint = std::numeric_limits<double>::max();

	inline void Reset() {
		feature = -1;
		gain = -std::numeric_limits<float>::infinity();
	}
	inline bool operator > (const SplitInfo& si) const {
		double local_gain = this->gain;
		double other_gain = si.gain;
		// replace nan with -inf
		if (local_gain == NAN) {
			local_gain = -std::numeric_limits<float>::infinity();;
		}
		// replace nan with -inf
		if (other_gain == NAN) {
			other_gain = -std::numeric_limits<float>::infinity();;
		}
		int local_feature = this->feature;
		int other_feature = si.feature;
		// replace -1 with max int
		if (local_feature == -1) {
			local_feature = INT32_MAX;
		}
		// replace -1 with max int
		if (other_feature == -1) {
			other_feature = INT32_MAX;
		}
		if (local_gain != other_gain) {
			return local_gain > other_gain;
		}
		else {
			// if same gain, use smaller feature
			return local_feature < other_feature;
		}
	}

	inline bool operator == (const SplitInfo& si) const {
		double local_gain = this->gain;
		double other_gain = si.gain;
		// replace nan with -inf
		if (local_gain == NAN) {
			local_gain = -std::numeric_limits<float>::infinity();
		}
		// replace nan with -inf
		if (other_gain == NAN) {
			other_gain = -std::numeric_limits<float>::infinity();
		}
		int local_feature = this->feature;
		int other_feature = si.feature;
		// replace -1 with max int
		if (local_feature == -1) {
			local_feature = INT32_MAX;
		}
		// replace -1 with max int
		if (other_feature == -1) {
			other_feature = INT32_MAX;
		}
		if (local_gain != other_gain) {
			return local_gain == other_gain;
		}
		else {
			// if same gain, use smaller feature
			return local_feature == other_feature;
		}
	}

};

}
