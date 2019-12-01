#pragma once

#include "dataset.h"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "data_partition.hpp"





namespace Tiny_LightGBM {

class TreeLearner {
	
public:
	static TreeLearner* CreateTreeLearner();

	virtual void Init(const Dataset* train_data) = 0;

};


class SerialTreeLearner :public TreeLearner {
public:
	void Init(const Dataset* train_data) override;

protected:

	const Dataset* train_data_;
	int num_data_;
	int num_features_;

	HistogramPool histogram_pool_;

	std::vector<SplitInfo> best_split_per_leaf_;

	std::vector<std::unique_ptr<OrderedBin>> ordered_bins_;

	bool has_ordered_bin_ = false;

	std::unique_ptr<LeafSplits> smaller_leaf_splits_;
	std::unique_ptr<LeafSplits> larger_leaf_splits_;

	std::unique_ptr<DataPartition> data_partition_;

	std::vector<int8_t> is_feature_used_;
	std::vector<int> valid_feature_indices_;

	std::vector<float> ordered_gradients_;
	std::vector<float> ordered_hessians_;


};



}