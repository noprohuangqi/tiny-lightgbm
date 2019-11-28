#include "dataset.h"
#include "bin.h"
#include "feature_group.h"


namespace Tiny_LightGBM {


std::vector<std::vector<int>> NoGroup(const std::vector<int>& used_features) {
	std::vector<std::vector<int>> features_in_group;
	features_in_group.resize(used_features.size());

	for (size_t i = 0; i < used_features.size(); ++i) {
		features_in_group[i].emplace_back(used_features[i]);
	}
	return features_in_group;
}

Dataset::Dataset() {}
Dataset::Dataset() {}

Dataset::Dataset(int num_data) {
	num_data_ = num_data;

	//??????
	group_bin_boundaries.push_back(0);
}


void Dataset::Construct(std::vector<std::unique_ptr<BinMapper>>& bin_mappers,
						int ** sample_non_zeros_indices,
						const int* num_per_col,
						int num_row) {

	num_total_features_ = static_cast<int>(bin_mappers.size());

	std::vector<int> used_features;
	for (int i = 0; i < static_cast<int>(bin_mappers.size()); ++i) {
		used_features.emplace_back(i);
	}

	//这里实际上很多工作是为了EFB展开的，先行忽略

	//将vector<int> 转换到 vector<vector<int>>
	//方便后续进行EFB的group操作
	auto feature_in_group = NoGroup(used_features);


	//这里省略掉EFB
	//


	//代表被选中的feature的全部数量，group*cnt_in_group
	//即可能？不是全部feature都会考虑计算
	num_features_ = 0;
	for (const auto& fs : feature_in_group) {
		num_features_ += static_cast<int>(fs.size());
	}

	int cur_fidx = 0;
	used_feature_map_ = std::vector<int>(num_total_features_, -1);
	real_feature_idx_.resize(num_features_);
	feature2group_.resize(num_features_);
	feature2subfeature_.resize(num_features_);
	num_groups_ = static_cast<int>(feature_in_group.size());
	for (int i = 0; i < num_groups_; ++i) {
		auto cur_features = feature_in_group[i];
		int cur_cnt_features = static_cast<int>(cur_features.size());

		std::vector<std::unique_ptr<BinMapper>> cur_bin_mappers;

		for (int j = 0; j < cur_cnt_features; ++j) {
			int real_fidx = cur_features[j];
			used_feature_map_[real_fidx] = cur_fidx;
			real_feature_idx_[cur_fidx] = real_fidx;
			feature2group_[cur_fidx] = i;
			feature2subfeature_[cur_fidx] = j;
			cur_bin_mappers.emplace_back(bin_mappers[real_fidx].release());
			++cur_fidx;
		}

		feature_groups_.emplace_back(
			std::unique_ptr<FeatureGroup>(
				new FeatureGroup(
					cur_cnt_features, cur_bin_mappers, num_row)));
	}

	feature_groups_.shrink_to_fit();
	group_bin_boundaries.clear();
	int num_total_bin = 0;

	for (int i = 0; i < num_groups_; ++i) {
		num_total_bin += feature_groups_[i]->num_total_bin_;
		group_bin_boundaries.push_back(num_total_bin);
	}
	int last_group = 0;
	group_feature_start_.reserve(num_groups_);
	group_feature_cnt_.reserve(num_groups_);
	group_feature_start_.push_back(0);
	group_feature_cnt_.push_back(1);

	for (int i = 1; i < num_features_; ++i) {
		const int group = feature2group_[i];
		if (group == last_group) {
			group_feature_cnt_.back() = group_feature_cnt_.back() + 1;
		}
		else {
			group_feature_start_.push_back(i);
			group_feature_cnt_.push_back(1);
			last_group = group;
		}
	}

}



bool Dataset::SetFloatField(const float* label) {


	metadata_.SetLabel(label , num_data_);



	return true;

}


}
