#pragma once



namespace Tiny_LightGBM {

class LeafSplits {
	
public:
	double min_constraint() const { return min_val_; }
	double max_constraint() const { return max_val_; }
	LeafSplits(int num_data) :num_data_in_leaf_(num_data), num_data_(num_data) ,data_indices_(nullptr){}

	void Init() {
		leaf_index_ = -1;
		data_indices_ = nullptr;
		num_data_in_leaf_ = 0;
		min_val_ = -std::numeric_limits<double>::max();
		max_val_ = std::numeric_limits<double>::max();


	}

	void Init(int leaf, const DataPartition* data_partition, const float* gradients, const float* hessians) {
		leaf_index_ = leaf;
		data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);

		double tmp_sum_gradients = 0.0f;
		double tmp_sum_hessians = 0.0f;
		for (int i = 0; i < num_data_in_leaf_; ++i) {
			int idx = data_indices_[i];
			tmp_sum_gradients += gradients[idx];
			tmp_sum_hessians += hessians[idx];
		}
		sum_gradients_ = tmp_sum_gradients;
		sum_hessians_ = tmp_sum_hessians;
		min_val_ = -std::numeric_limits<double>::max();
		max_val_ = std::numeric_limits<double>::max();



	}

	void Init(int leaf, const DataPartition* data_partition, double sum_gradients, double sum_hessians) {
		leaf_index_ = leaf;
		data_indices_ = data_partition->GetIndexOnLeaf(leaf, &num_data_in_leaf_);
		sum_gradients_ = sum_gradients;
		sum_hessians_ = sum_hessians;
		min_val_ = -std::numeric_limits<double>::max();
		max_val_ = std::numeric_limits<double>::max();
	}


	const int* data_indices() const { return data_indices_; }
	int num_data_in_leaf() const { return num_data_in_leaf_; }
	int LeafIndex() const { return leaf_index_; }
	double sum_gradients() const { return sum_gradients_; }
	double sum_hessians() const { return sum_hessians_; }

	void SetValueConstraint(double min, double max) {
		min_val_ = min;
		max_val_ = max;
	}
private:
	int num_data_in_leaf_;
	int num_data_;
	int leaf_index_;

	const int* data_indices_;

	double sum_gradients_;
	double sum_hessians_;
	double min_val_;
	double max_val_;
};



}