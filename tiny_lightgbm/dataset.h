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


class Metadata {

public:

	void SetLabel(const float* label , int len);

	inline const float* label() const { return label_.data(); }

private:
	std::vector<float> label_;

};

class Dataset {
public:

	

	TINY_LIGHTGBM_EXPORT Dataset();

	TINY_LIGHTGBM_EXPORT Dataset(int num_data);

	TINY_LIGHTGBM_EXPORT Dataset();


	void Construct(std::vector<std::unique_ptr<BinMapper>>& bin_mappers , 
					int ** sample_non_zeros_indices , 
					const int* num_per_col,
					int nuim_row);

	void ConstructHistograms(const std::vector<int>& is_feature_used,
							const int* data_indices, int num_data,
							int leaf_idx,
							std::vector<std::unique_ptr<OrderedBin>>& ordered_bins,
							const float* gradients , const float* hessians,
							float* ordered_gradients, float* ordered_hessians,
							bool is_constant_hessian,
							HistogramBinEntry* hist_data) const;

	inline void PushOneRow(int row_idx, const std::vector<double>& feature_values) {

		for (size_t i = 0; i < feature_values.size(); ++i) {
			int feature_idx = used_feature_map_[i];

			const int group = feature2group_[feature_idx];
			//更新值到某一group的顺序
			const int sub_feature = feature2subfeature_[feature_idx];

			feature_groups_[group]->PushData(sub_feature, row_idx, feature_values[i]);
		}
	}

	bool SetFloatField(const float* label);


	inline const Metadata& metadata() const { return metadata_; }
	inline int num_data() const { return num_data_; }

	inline int num_features() const { return num_features_; }

	inline int FeatureNumBin(int i) const {
		const int group = feature2group_[i];
		const int sub_feature = feature2subfeature_[i];

		return feature_groups_[group]->bin_mappers_[sub_feature]->num_bin();

	}

	inline const BinMapper* FeatureBinMapper(int i) const {
		const int group = feature2group_[i];
		const int sub_feature = feature2subfeature_[i];

		return feature_groups_[group]->bin_mappers_[sub_feature].get();
	}

	inline int NumTotalBin() const {
		return group_bin_boundaries_.back();
	}

	inline int SubFeatureBinOffset(int i) const {
		const int sub_feature = feature2subfeature_[i];

		if (sub_feature == 0) {
			return 1;
		}
		else {
			return 0;
		}
	}

	inline void CreateOrderedBins(std::vector<std::unique_ptr<OrderedBin>>* ordered_bins ) const {

		ordered_bins->resize(num_groups_);
		for (int i = 0; i < num_groups_; ++i) {
			ordered_bins->at(i).reset(feature_groups_[i]->bin_data_->CreateOrderedBin());
		}
	}

	inline std::vector<int> ValidFeatureIndices() const {
		std::vector<int> ret;
		for (int i = 0; i < num_total_features_; ++i) {
			if (used_feature_map_[i] >= 0) {
				ret.push_back(i);
			}
		}
		return ret;

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

	std::vector<int> group_bin_boundaries_;

	std::vector<int> group_feature_start_;
	std::vector<int> group_feature_cnt_;


	Metadata metadata_;


};







}
