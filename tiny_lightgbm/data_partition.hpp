#pragma once

#include<algorithm>

namespace Tiny_LightGBM {

class DataPartition {

public:
	DataPartition(int num_data, int num_leaves):num_data_(num_data),num_leaves_(num_leaves) {


	}
	void Init() {
		std::fill(leaf_begin_.begin(), leaf_begin_.end(), 0);
		std::fill(leaf_count_.begin(), leaf_count_.end(), 0);

		leaf_count_[0] = num_data_;
		for (int i = 0; i < num_data_; ++i) {
			indices_[i] = i;
		}

	}

	int leaf_count(int leaf_idx) const { return leaf_count_[leaf]; }

	const int* GetIndexOnLeaf(int leaf, int* out_len) const {

		int begin = leaf_begin_[leaf];
		*out_len = leaf_count_[leaf];
		return indices_.data() + begin;

	}


private:
	int num_data_;
	int num_leaves_;

	std::vector<int> leaf_begin_;
	std::vector<int> leaf_count_;
	std::vector<int> indices_;
	std::vector<int> temp_left_indices_;
	std::vector<int> temp_right_indices_;

};


}
