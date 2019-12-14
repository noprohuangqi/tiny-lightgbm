#pragma once

#include <vector>
#include <functional>
#include <unordered_map>
#include <sstream>
namespace Tiny_LightGBM {



struct  HistogramBinEntry{
public:
	double sum_gradients = 0.0f;
	double sum_hessians = 0.0f;
	int cnt = 0;
};


//实现将原始feature分布进行装桶，bins = 255
//BinMapper是针对单个原始feature的
class BinMapper {

public:
	BinMapper();
	~BinMapper();

	//BinMapper的核心函数，构造bin
	void FindBin(double* values, int num_values, int num_row);

	inline int num_bin() const { return num_bin_; }

	inline int ValueToBin(double value) const;

	inline int GetDefaultBin() const {
		return default_bin_;
	}

	inline double BinToValue(uint32_t bin) const {
		
		//默认是在numerical的情况下
		return bin_upper_bound_[bin];
		
	}
	
private:
	int num_bin_;
	double min_val_;
	double max_val_;
	int default_bin_;
	std::vector<double> bin_upper_bound_;

};

inline int BinMapper::ValueToBin(double value) const {
	int l = 0;
	int r = num_bin_ - 1;
	while (l < r) {
		int m = (r + l - 1) / 2;
		if (value <= bin_upper_bound_[m]) {
			r = m;
		}
		else {
			l = m + 1;
		}
	}
	return l;

}

class OrderedBin {

public:
private:

};


class Bin {

public:
	static Bin* CreateBin(int num_data, int num_bin);


	virtual void Push(int idx, int value) = 0;

	virtual OrderedBin* CreateOrderedBin() const = 0;
	virtual void ConstructHistogram(const int* data_indices, int num_data, const float* ordered_gradients, HistogramBinEntry* out) const;
private:


};




class DenseBin :public Bin {
public:
	//DenseBin初始化非常重要的地方，就是把data_的所有数据都初始化为0
	//data_是一个vector，其index代表数据的index，value代表某个feature的bin
	//即所有feature默认都是放在bin0这个地方
	DenseBin(int num_data):num_data_(num_data),data_(num_data_ , 0){}


	void Push(int idx, int value) override{
		data_[idx] = value;
	}

	//这里是说，dense的情况下就不需要排序了
	OrderedBin* CreateOrderedBin() const override { return nullptr; }

	void ConstructHistogram(const int* data_indices, int num_data, const float* ordered_gradients, HistogramBinEntry* out) const override;

	int Split(
		uint32_t min_bin, uint32_t max_bin, uint32_t default_bin, bool default_left,
		uint32_t threshold, int* data_indices, int num_data,
		int* lte_indices, int* gt_indices) const;


private:

	int num_data_;
	std::vector<double> data_;
};



}

