#pragma once

#include "bin.h"
#include "split_info.hpp"
#include  "utils.h"

namespace Tiny_LightGBM {

class FeatureMetainfo {

public:
	int num_bin;
	int default_bin;

	int bias = 0;
};

class FeatureHistogram {
public:
	void Init(HistogramBinEntry* data, const FeatureMetainfo* meta) {
		meta_ = meta;
		data_ = data;

		find_best_threshold_fun_ = std::bind(&FeatureHistogram::FindBestThresholdNumerical , this ,
											std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, 
											std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);

	}

	static double ThresholdL1(double s, double l1) {
		const double reg_s = std::max(0.0, std::fabs(s) - l1);
		return Utils::Sign(s) *reg_s;
	}

	static double CalculateSplittedLeafOutput(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {
		double ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2);
		return ret;
	}

	static double GetLeafSplitGain(double sum_gradients, double sum_hessians, double l1, double l2, double max_delta_step) {

		double output = CalculateSplittedLeafOutput(sum_gradients, sum_hessians, l1, l2, max_delta_step);
		return GetLeafSplitGainGivenOutput(sum_gradients, sum_hessians, l1, l2, output);

	}

	static double GetLeafSplitGainGivenOutput(double sum_gradients, double sum_hessians, double l1, double l2, double output) {

		const double sg_l1 = ThresholdL1(sum_gradients, l1);
		return -(2.0 * sg_l1 * output + (sum_hessians + l2)*output*output);
	}

	void FindBsetThresholdSequence(double sum_gradient , double sum_hessian , int num_data ,
									double min_constraint , double max_constraint,
									double min_gain_shift , SplitInfo* output , 
									int dir , bool skip_default_bin , bool use_na_sa_missing) {



	}

	void FindBestThresholdNumerical(double sum_gradient , double sum_hessian , 
									int num_data , double min_constraint , 
									double max_constraint , SplitInfo* output) {

		is_splittable_ = false;
		//参数分别是l1 ， l2 ， max_delta_step
		double gain_shift = GetLeafSplitGain(sum_gradient, sum_hessian, 0.0, 0.0, 0.0);
		//double min_gain_to_split = 0.0;
		double min_gain_shift = gain_shift + 0.0;
		FindBsetThresholdSequence(sum_gradient, sum_hessian, num_data, min_constraint, max_constraint, min_gain_shift, output, -1, false, false);


	}

private:

	bool is_splittable_;
	const FeatureMetainfo* meta_;
	HistogramBinEntry* data_;

	std::function<void(double, double, int, double, double, SplitInfo*)> find_best_threshold_fun_;
};



class HistogramPool {
public:
	HistogramPool() {
		cache_size_ = 0;
		total_size_ = 0;
	}

	void DynamicChangeSize(const Dataset* train_data, int cache_size) {
		if (feature_metas_.empty()) {
			int num_feature = train_data->num_features();
			feature_metas_.resize(num_feature);

			for (int i = 0; i < num_feature; ++i) {

				feature_metas_[i].num_bin = train_data->FeatureNumBin(i);

				feature_metas_[i].default_bin = train_data->FeatureBinMapper(i)->GetDefaultBin();
				if (train_data->FeatureBinMapper(i)->GetDefaultBin() == 0) {
					feature_metas_[i].bias = 1;
				}
				else {
					feature_metas_[i].bias = 0;
				}
			}
			

		}
		int num_total_bin = train_data->NumTotalBin();

		int old_cache_size = static_cast<int>(pool_.size());

		if (cache_size > old_cache_size) {
			pool_.resize(cache_size);
			data_.resize(cache_size);
		}
		

		for (int i = old_cache_size; i < cache_size; ++i) {

			pool_[i].reset(new FeatureHistogram[train_data->num_features()]);

			data_[i].resize(num_total_bin);

			int offset = 0;

			for (int j = 0; j < train_data->num_features(); ++j) {
				offset += static_cast<int>(train_data->SubFeatureBinOffset(j));
				pool_[i][j].Init(data_[i].data() + offset, &feature_metas_[j]);

				//又有减一的操作
				auto num_bin = train_data->FeatureNumBin[j];
				if (train_data->FeatureBinMapper(j)->GetDefaultBin() == 0) {
					num_bin -= 1;
				}

				offset += static_cast<int>(num_bin);

			}



		}


	}


private:
	int cache_size_;
	int total_size_;

	std::vector<FeatureMetainfo> feature_metas_;

	std::vector<std::unique_ptr<FeatureHistogram[]>> pool_;

	std::vector<std::vector<HistogramBinEntry>> data_;

};


}
