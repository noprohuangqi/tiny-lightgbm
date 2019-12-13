#pragma once

#include "dataset.h"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "data_partition.hpp"
#include "tree.h"





namespace Tiny_LightGBM {

class TreeLearner {
	
public:
	static TreeLearner* CreateTreeLearner();

	virtual void Init(const Dataset* train_data) = 0;
	virtual Tree* Train(const float* gradients, const float* hessians, bool is_constant_hessian) = 0;
	
	virtual void AddPredictionToScore(const Tree* tree, double* out_score) const = 0;
};


class SerialTreeLearner :public TreeLearner {
public:
	void Init(const Dataset* train_data) override;

protected:

	virtual void ConstructHistograms(const std::vector<int>& is_feature_used, bool use_substract);
	virtual void FindBestSplitsFromHistograms(const std::vector<int>& is_feature_used, bool use_subtract);
	virtual void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf);

	void AddPredictionToScore(const Tree* tree, double* out_score) const override {
		if (tree->num_leaves() <= 1) { return; }
		

		for (int i = 0; i < tree->num_leaves(); ++i) {
			double output = static_cast<double>(tree->LeafOutput(i));
			int cnt_leaf_data = 0;
			auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
			for (int j = 0; j < cnt_leaf_data; ++j) {
				out_score[tmp_idx[j]] += output;
			}
		}
	}


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

	FeatureHistogram* smaller_leaf_histogram_array_;
	FeatureHistogram* larger_leaf_histogram_array_;
	FeatureHistogram* parent_leaf_histogram_array_;


	const float* gradients_;
	const float* hessians_;
	bool is_constant_hessian_;

	Tree* Train(const float* gradients , const float* hessians , bool is_constant_hessian) override;

	virtual void BeforeTrain();

	virtual bool BeforeFindBsetSplit(const Tree* tree, int left_leaf, int right_leaf);

	inline virtual int GetGlobalDataCountInLeaf(int leaf_idx) const;

	virtual void FindBestSplits();

};

inline int SerialTreeLearner::GetGlobalDataCountInLeaf(int leaf_idx) const {

	if (leaf_idx >= 0) {
		return data_partition_->leaf_count(leaf_idx);
	}
	else {
		return 0;
	}
}



}