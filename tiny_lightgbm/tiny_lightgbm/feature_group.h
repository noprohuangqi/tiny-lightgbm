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


	//没有sparse的考虑
	FeatureGroup(int num_feature , 
				std::vector<std::unique_ptr<BinMapper>>& bin_mappers , 
				int num_data 
				):num_feature_(num_feature) {
		
		//假设featuregroup只有一个feature，该feature假设有10个bin。
		//这里需要先增加一个bin到0位置。
		num_total_bin_ = 1;
		bin_offsets_.emplace_back(num_total_bin_);

		
		for (int i = 0; i < num_feature; ++i) {
			bin_mappers_.emplace_back(bin_mappers[i].release());
			auto num_bin = bin_mappers_[i]->num_bin();
			
			//骚操作
			//这里可以看到，不同的feature进行处理时是不一样的。
			//如果原始feature的BinMapper第一个bin就是value 0所在的
			//那么其num_bin -= 1
			if (bin_mappers_[i]->GetDefaultBin() == 0) {
				num_bin -= 1;
			}
			//如果不止一个feature的话，可以看到不同feature的bin的个数得到了叠加
			//例如feature1的bin是30个，feature2的bin是40个。
			//那么feature1就占据1-30的位置，feature2就占据30-70（如果两个feature的default bin都不是0）
			//而0这个位置就空余了出来，后续会提及怎么使用
			num_total_bin_ += num_bin;
			bin_offsets_.emplace_back(num_total_bin_);

		}

		//默认所有的feature都是密集的，DenseBin
		bin_data_.reset(Bin::CreateBin(num_data, num_total_bin_));


	}


	inline void PushData(int sub_feature_idx, int line_idx, double value) {

		//得到当前这个feature应该填充到哪个bin里面去
		int bin = bin_mappers_[sub_feature_idx]->ValueToBin(value);

		//如果value是0，即应该填充到default bin的位置，放弃。
		//因为本来所有的数据初始化就是放在bin 0的位置，这个操作在DenseBin的构造里面
		//需要注意的是，如果是EFB的情况下，假设第一个和第二个feature都遇到了value0
		//那么都应该放在bin0，没有区别了
		if (bin == bin_mappers_[sub_feature_idx]->GetDefaultBin()) { return; }

		//bin_offsets的值是[1,31,71]。注意最开始是1，不是0
		bin += bin_offsets_[sub_feature_idx];

		//假设当前feature group只有一个feature，而且default bin =0
		//假设进来一个value=0，那么在之前就会return
		//假设进来一个value>0，那么在这里bin-=1
		//如果不减的话，bin 1就会没有任何数据在里面（因为bin += bin_offsets之后，至少都是2）

		//再来假设当前feature group有多个feature，而且某一个feature的default bin也是0
		//那么也是一样的考虑

		//这里的操作虽然复杂，核心思想是：
		//多个feature作为一个group的情况下，每个feature的value 0 都应该放在一起（bin 0位置）
		//而不是作为不同的bin
		if (bin_mappers_[sub_feature_idx]->GetDefaultBin() == 0) {
			bin -= 1;

		}
		//填数据
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
