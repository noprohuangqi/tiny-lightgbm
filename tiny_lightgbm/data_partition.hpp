#pragma once

#include<algorithm>

namespace Tiny_LightGBM {

class DataPartition {

public:
	DataPartition(int num_data, int num_leaves):num_data_(num_data),num_leaves_(num_leaves) {


	}
	void Init() {

		num_threads_ = 1;

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

	void Split(int leaf, const Dataset* dataset, int feature, const uint32_t* threshold, int num_threshold, bool default_left, int right_leaf) {

		const int min_inner_size  = 512;
		const int begin = leaf_begin_[leaf];
		const int cnt = leaf_count_[leaf];

		left_cnts_buf_[0] = 0;
		right_cnts_buf_[0] = 0;
		int cur_start = 0;
		int cur_cnt = cnt;

		int cur_left_count = dataset->Split(feature, threshold, num_threshold, default_left, indices_.data() + begin + cur_start, cur_cnt,
			temp_left_indices_.data() + cur_start, temp_right_indices_.data() + cur_start);
		offsets_buf_[0] = cur_start;
		left_cnts_buf_[0] = cur_left_count;
		right_cnts_buf_[0] = cur_cnt - cur_left_count;

		int left_cnt = left_cnts_buf_[0];
		left_write_pos_buf_[0] = 0;
		right_write_pos_buf_[0] = 0;
		if (left_cnts_buf_[0] > 0) {
			std::memcpy(indices_.data() + begin + left_write_pos_buf_[0],
				temp_left_indices_.data() + offsets_buf_[0], left_cnts_buf_[0] * sizeof(int));
		}
		if (right_cnts_buf_[0] > 0) {
			std::memcpy(indices_.data() + begin + left_cnt + right_write_pos_buf_[0],
				temp_right_indices_.data() + offsets_buf_[0], right_cnts_buf_[0] * sizeof(int));
		}
		leaf_count_[leaf] = left_cnt;
		leaf_begin_[right_leaf] = left_cnt + begin;
		leaf_count_[right_leaf] = cnt - left_cnt;
		
	}



private:
	int num_data_;
	int num_leaves_;

	std::vector<int> leaf_begin_;
	std::vector<int> leaf_count_;
	std::vector<int> indices_;
	std::vector<int> temp_left_indices_;
	std::vector<int> temp_right_indices_;
	int num_threads_;
	/*! \brief Buffer for multi-threading data partition, used to store offset for different threads */
	std::vector<int> offsets_buf_;
	/*! \brief Buffer for multi-threading data partition, used to store left count after split for different threads */
	std::vector<int> left_cnts_buf_;
	/*! \brief Buffer for multi-threading data partition, used to store right count after split for different threads */
	std::vector<int> right_cnts_buf_;
	/*! \brief Buffer for multi-threading data partition, used to store write position of left leaf for different threads */
	std::vector<int> left_write_pos_buf_;
	/*! \brief Buffer for multi-threading data partition, used to store write position of right leaf for different threads */
	std::vector<int> right_write_pos_buf_;
};


}
