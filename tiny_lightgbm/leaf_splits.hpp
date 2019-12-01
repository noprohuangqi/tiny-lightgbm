#pragma once



namespace Tiny_LightGBM {

class LeafSplits {
	
public:

	LeafSplits(int num_data) :num_data_in_leaf_(num_data), num_data_(num_data) ,data_indices_(nullptr){}

private:
	int num_data_in_leaf_;
	int num_data_;
	int leaf_index_;

	const int* data_indices_;

};



}