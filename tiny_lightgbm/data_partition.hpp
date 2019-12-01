#pragma once



namespace Tiny_LightGBM {

class DataPartition {

public:
	DataPartition(int num_data, int num_leaves):num_data_(num_data),num_leaves_(num_leaves) {


	}

private:
	int num_data_;
	int num_leaves_;

};


}
