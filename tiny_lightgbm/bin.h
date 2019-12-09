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
class BinMapper {

public:
	BinMapper();
	~BinMapper();


	void FindBin(double* values, int num_values, int num_row);

	inline int num_bin() const { return num_bin_; }

	inline int ValueToBin(double value) const;

	inline int GetDefaultBin() const {
		return default_bin_;
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

private:


};




class DenseBin :public Bin {
public:
	DenseBin(int num_data):num_data_(num_data),data_(num_data_ , 0){}


	void Push(int idx, int value) override{
		data_[idx] = value;
	}

	//这里是说，dense的情况下就不需要排序了
	OrderedBin* CreateOrderedBin() const override { return nullptr; }

private:

	int num_data_;
	std::vector<double> data_;
};



}

