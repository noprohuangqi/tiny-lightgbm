#pragma once

#include "define.h"
#include "bin.h"
#include "feature_group.h"


#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <unordered_set>
#include <mutex>

namespace Tiny_LightGBM {

class Dataset {
public:

	

	TINY_LIGHTGBM_EXPORT Dataset();

	TINY_LIGHTGBM_EXPORT Dataset(int num_data);

	TINY_LIGHTGBM_EXPORT Dataset();


	void Construct(std::vector<std::unique_ptr<BinMapper>>& bin_mappers , 
					int ** sample_non_zeros_indices , 
					const int* num_per_col,
					int nuim_row);

	inline void PushOneRow(int row_idx, const std::vector<double>& feature_values) {

		for (size_t i = 0; i < feature_values.size(); ++i) {
			int feature_idx = used_feature_map_[i];

			const int group = feature2group_[feature_idx];
			//更新值到某一group的顺序
			const int sub_feature = feature2subfeature_[feature_idx];

			feature
		}
	}


private:

	int num_data_;
	int num_total_features_;
	int num_features_;
	int num_groups_;
	std::vector<int> used_feature_map_;
	std::vector<int> real_feature_idx_;
	std::vector<int> feature2group_;
	std::vector<int> feature2subfeature_;

	std::vector<std::unique_ptr<FeatureGroup>> feature_groups_;

	std::vector<int> group_bin_boundaries;

	std::vector<int> group_feature_start_;
	std::vector<int> group_feature_cnt_;



};







}
