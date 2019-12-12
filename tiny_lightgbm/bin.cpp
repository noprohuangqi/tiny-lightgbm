#include "bin.h"
#include "utils.h"
#include "config.h"

#include <cmath>
#include <cstring>
#include <cstdint>

#include <limits>
#include <vector>
#include <algorithm>

namespace Tiny_LightGBM {

	BinMapper::BinMapper() {}
	BinMapper::~BinMapper() {}


	std::vector<double> GreedyFindBin(const double* distinct_values ,const int* counts , int num_distinct_values,
									int max_bin , 
									int total_cnt , int min_data_in_bin) {

		std::vector<double> bin_upper_bound;

		if (num_distinct_values <= max_bin) {

			bin_upper_bound.clear();

			int cur_cnt_inbin = 0;
			for (int i = 0; i < num_distinct_values - 1; ++i) {
				cur_cnt_inbin += counts[i];

				if (cur_cnt_inbin > min_data_in_bin) {
					auto val = Utils::GetDoubleUpperBound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
					if (bin_upper_bound.empty() || !Utils::CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
						bin_upper_bound.push_back(val);
						cur_cnt_inbin = 0;
					}
				}
			}
			cur_cnt_inbin += counts[num_distinct_values - 1];
			bin_upper_bound.push_back(std::numeric_limits<double>::infinity());

		}
		else {
			//默认是3
			if (min_data_in_bin > 0) {
				max_bin = std::min(max_bin, static_cast<int>(total_cnt / min_data_in_bin));
				max_bin = std::max(max_bin, 1);
			}

			double mean_bin_size = static_cast<double>(total_cnt) / max_bin;


			int rest_bin_cnt = max_bin;
			int rest_sample_cnt = static_cast<int>(total_cnt);
			std::vector<bool> is_big_count_value(num_distinct_values, false);

			//先遍历一次，找到单个value的量就足够形成一个bin的位置
			for (int i = 0; i < num_distinct_values; ++i) {

				if (counts[i] > mean_bin_size) {
					is_big_count_value[i] = true;
					--rest_bin_cnt;
					rest_sample_cnt -= counts[i];
				}
			}
			mean_bin_size = static_cast<double>(rest_sample_cnt) / rest_bin_cnt;
			std::vector<double> upper_bounds(max_bin, std::numeric_limits<double>::infinity());
			std::vector<double> lower_bounds(max_bin, std::numeric_limits<double>::infinity());

			int bin_cnt = 0;
			lower_bounds[bin_cnt] = distinct_values[0];

			int cur_cnt_inbin = 0;

			//开始划分，考虑两种情况
			//一是单个value即可组成一个bin
			//二是多个value组成一个bin
			//当然也有可能，当前还不够一个bin，但下一个就可以单独成bin
			for (int i = 0; i < num_distinct_values - 1; ++i) {
				if (!is_big_count_value[i]) {
					rest_sample_cnt -= counts[i];

				}
				cur_cnt_inbin += counts[i];
				if (is_big_count_value[i] || cur_cnt_inbin >= mean_bin_size || 
					(is_big_count_value[i+1] && cur_cnt_inbin >= std::max(1.0 , mean_bin_size*0.5f))) {

					upper_bounds[bin_cnt] = distinct_values[i];
					++bin_cnt;
					lower_bounds[bin_cnt] = distinct_values[i + 1];

					if (bin_cnt >= max_bin - 1) { break; }
					cur_cnt_inbin = 0;

					if (!is_big_count_value[i]) {
						--rest_bin_cnt;
						mean_bin_size = rest_sample_cnt / static_cast<double>(rest_bin_cnt);
					}
				}
			}
			bin_upper_bound.clear();

			for (int i = 0; i < bin_cnt; ++i) {
				auto val = Utils::GetDoubleUpperBound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
				if (bin_upper_bound.empty() || !Utils::CheckDoubleEqualOrdered(bin_upper_bound.back(), val)) {
					bin_upper_bound.push_back(val);
				}
			}

			bin_upper_bound.push_back(std::numeric_limits<double>::infinity());

		}
		return bin_upper_bound;
	}


	std::vector<double> FindBinWithZeroAsOneBin(const double* distinct_values , const int* counts,
												int num_distinct_values,
												int max_bin , int min_data_in_bin , int num_row) {

		std::vector<double> bin_upper_bound;

		//统计正负0的量
		int left_cnt_data = 0;
		int cnt_zero = 0;
		int right_cnt_data = 0;
		for (int i = 0; i < num_distinct_values; ++i) {

			if (distinct_values[i] <= -kZeroThreshold) {
				left_cnt_data += counts[i];
			}
			else if (distinct_values[i] > kZeroThreshold) {
				right_cnt_data += counts[i];
			}
			else {
				cnt_zero += counts[i];
			}
		}

		//寻找正负0的位置
		int left_cnt = -1;
		for (int i = 0; i < num_distinct_values; ++i) {
			if (distinct_values[i] > -kZeroThreshold) {
				left_cnt = i;
				break;
			}
		}

		if (left_cnt < 0) {
			left_cnt = num_distinct_values;
		}

		//如果有负数
		if (left_cnt > 0) {
			int left_max_bin = static_cast<int>(static_cast<double>(left_cnt_data) / (num_row - cnt_zero) * (max_bin - 1));
			left_max_bin = std::max(1, left_max_bin);
			bin_upper_bound = GreedyFindBin(distinct_values, counts, left_cnt, left_max_bin, left_cnt_data, min_data_in_bin);
			bin_upper_bound.back() = -kZeroThreshold;
		}

		int right_start = -1;
		for (int i = left_cnt; i < num_distinct_values; ++i) {
			if (distinct_values[i] > kZeroThreshold) {
				right_start = i;
				break;
			}
		}
		//如果有正数bin
		if (right_start >= 0) {
			int right_max_bin = max_bin - 1 - static_cast<int>(bin_upper_bound.size());
			auto right_bounds = GreedyFindBin(distinct_values + right_start, counts + right_start,
				num_distinct_values - right_start, right_max_bin, right_cnt_data, min_data_in_bin);
			bin_upper_bound.push_back(kZeroThreshold);
			bin_upper_bound.insert(bin_upper_bound.end(), right_bounds.begin(), right_bounds.end());
		}
		else {
			bin_upper_bound.push_back(std::numeric_limits<double>::infinity());
		}
		return bin_upper_bound;



	}


	void BinMapper::FindBin(double* values, int num_values, int num_row) {
		//num_values 是除了0以外的元素量
		//明确有三种类型的数据，nan ， 0和其他

		int tmp_num_sample_values = 0;
		for (int i = 0; i < num_values; ++i) {
			if (!std::isnan(values[i])) {
				values[tmp_num_sample_values++] = values[i];
			}
		}
		//na_cnt
		int na_cnt = static_cast<int>(num_values - tmp_num_sample_values);
		//zero_cnt
		int zero_cnt = static_cast<int>(num_row - num_values - na_cnt);
		//num_values 即其他值
		num_values = tmp_num_sample_values;

		default_bin_ = 0;

		//只对其他值进行排序
		std::stable_sort(values, values + num_values);

		std::vector<double> distinct_values;
		std::vector<int> counts;


		// 如果没有正常值，或者第一个值是正数
		if (num_values == 0 || (values[0] > 0.0f && zero_cnt > 0)) {
			distinct_values.push_back(0.0f);
			counts.push_back(zero_cnt);
		}

		//push第一个进去
		if (num_values > 0) {
			distinct_values.push_back(values[0]);
			counts.push_back(1);
		}

		for (int i = 1; i < num_values; ++i) {
			//如果前后不等
			if (!Utils::CheckDoubleEqualOrdered(values[i - 1], values[i])) {
				//前负后正，插入0
				if (values[i - 1] <0.0f && values[i] > 0.0f) {
					distinct_values.push_back(0.0f);
					counts.push_back(zero_cnt);
				}
				distinct_values.push_back(values[i]);
				counts.push_back(1);
			}
			else {
				distinct_values.back() = values[i];
				++counts.back();
			}
		}

		//后插0，如果所有的值都是负的
		if (num_values > 0 && values[num_values - 1] < 0.0f && zero_cnt > 0) {
			distinct_values.push_back(0.0f);
			counts.push_back(zero_cnt);
		}


		min_val_ = distinct_values.front();
		max_val_ = distinct_values.back();
		std::vector<int> cnt_in_bin;
		int num_distinct_values = static_cast<int>(distinct_values.size());



		bin_upper_bound_ = FindBinWithZeroAsOneBin(distinct_values.data(),
													counts.data(), num_distinct_values,
													Config::max_bin , Config::min_data_in_bin,num_row);

		num_bin_ = static_cast<int>(bin_upper_bound_.size());

		//单独的大括号会限制里面新定义的变量的寿命
		{
			cnt_in_bin.resize(num_bin_, 0);
			int i_bin = 0;
			for (int i = 0; i < num_distinct_values; ++i) {
				if (distinct_values[i] > bin_upper_bound_[i_bin]) {
					++i_bin;
				}
				cnt_in_bin[i_bin] += counts[i];
			}
		}

		default_bin_ = ValueToBin(0);

	}

	//
	Bin* Bin::CreateBin(int num_data, int num_bin) {

		return new DenseBin(num_data);

	}


	void DenseBin::ConstructHistogram(const int* data_indices, int num_data, const float* ordered_gradients, HistogramBinEntry* out) const override{

		for (int i = 0; i < num_data; ++i) {
			const int bin = data_[data_indices[i]];
			out[bin].sum_gradients += ordered_gradients[i];
			++out[bin].cnt;
		}

	}

	int DenseBin::Split(uint32_t min_bin, uint32_t max_bin, uint32_t default_bin, bool default_left,
						uint32_t threshold, int* data_indices, int num_data,
						int* lte_indices, int* gt_indices) const {
		if (num_data <= 0) { return 0; }
		int th = static_cast<int>(threshold + min_bin);
		const int minb = static_cast<int>(min_bin);
		const int maxb = static_cast<int>(max_bin);
		int t_default_bin = static_cast<int>(min_bin + default_bin);
		if (default_bin == 0) {
			th -= 1;
			t_default_bin -= 1;
		}
		int lte_count = 0;
		int gt_count = 0;
		int* default_indices = gt_indices;
		int* default_count = &gt_count;

		for (int i = 0; i < num_data; ++i) {
			const int idx = data_indices[i];
			const int bin = data_[idx];
			if (bin < minb || bin > maxb || t_default_bin == bin) {
				default_indices[(*default_count)++] = idx;
			}
			else if (bin > th) {
				gt_indices[gt_count++] = idx;
			}
			else {
				lte_indices[lte_count++] = idx;
			}
		}


	}

}
