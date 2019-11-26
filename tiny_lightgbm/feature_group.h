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


private:
	std::vector<int> bin_offsets_;
	int num_total_bin_;
	int num_feature_;
	std::vector<std::unique_ptr<BinMapper>> bin_mappers_;

	std::unique_ptr<Bin> bin_data_;


};

}
