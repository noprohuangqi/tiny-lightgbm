#pragma once

#include "bin.h"



#include <cstdio>
#include <memory>
#include <vector>



namespace Tiny_LightGBM {

class Dataset;
class FeatureGroup {

public:

	friend Dataset;


	//Ã»ÓÐsparseµÄ¿¼ÂÇ
	FeatureGroup(int num_feature , 
				std::vector<std::unique_ptr<BinMapper>>& bin_mappers , 
				int num_data 
				):num_feature_(num_feature) {
		//???
		num_total_bin_ = 1;

		bin_offsets_.emplace_back(num_total_bin_);

		int cnt_non_zero = 0;
		//???
		for (int i = 0; i < num_feature; ++i) {
			bin_mappers_.emplace_back(bin_mappers[i].release());
			auto num_bin = bin_mappers_[i]->num_bin();
			num_total_bin_ += num_bin;
			bin_offsets_.emplace_back(num_total_bin_);

		}

		bin_data_.reset(Bin::CreateBin(num_data, num_total_bin_));


	}


	inline void PushData(int sub_feature_idx, int line_idx, double value) {
		int bin = bin_mappers_[sub_feature_idx]->ValueToBin(value);

		// ???????????????????????????
		if (bin == bin_mappers_[sub_feature_idx]->GetDefaultBin()) { return; }

		bin += bin_offsets_[sub_feature_idx];
		if (bin_mappers_[sub_feature_idx]->GetDefaultBin() == 0) {
			bin -= 1;

		}
		bin_data_->Push(line_idx, bin);


	}

	inline int Split(
		int sub_feature,
		const uint32_t* threshold,
		int num_threshold,
		bool default_left,
		int* data_indices, int num_data,
		int* lte_indices, int* gt_indices) const {

		uint32_t min_bin = bin_offsets_[sub_feature];
		uint32_t max_bin = bin_offsets_[sub_feature + 1] - 1;
		uint32_t default_bin = bin_mappers_[sub_feature]->GetDefaultBin();
		
			
		return bin_data_->Split(min_bin, max_bin, default_bin, default_left,
			*threshold, data_indices, num_data, lte_indices, gt_indices);
		
	}


private:
	std::vector<int> bin_offsets_;
	int num_total_bin_;
	int num_feature_;
	std::vector<std::unique_ptr<BinMapper>> bin_mappers_;

	std::unique_ptr<Bin> bin_data_;


};

}
