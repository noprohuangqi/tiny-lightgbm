

#include "tree_learner.h"
#include "config.h"

namespace Tiny_LightGBM {


void SerialTreeLearner::Init(const Dataset* train_data) {
	
	train_data_ = train_data;
	num_data_ = train_data_->num_data();
	num_features_ = train_data_->num_features();

	//Ä¬ÈÏÊÇ31
	int max_cache_size = Config::num_leaves;

	histogram_pool_.DynamicChangeSize(train_data_,max_cache_size);

	best_split_per_leaf_.resize(max_cache_size);

	train_data_->CreateOrderedBins(&ordered_bins_);
	for (int i = 0; i < static_cast<int>(ordered_bins_.size()); ++i) {
		if (ordered_bins_[i] != nullptr) {
			has_ordered_bin_ = true;
			break;
		}
	}
	smaller_leaf_splits_.reset(new LeafSplits(train_data_->num_data()));
	larger_leaf_splits_.reset(new LeafSplits(train_data_->num_data()));

	data_partition_.reset(new DataPartition(num_data_, max_cache_size));
	is_feature_used_.resize(num_features_);

	valid_feature_indices_ = train_data_->ValidFeatureIndices();

	ordered_gradients_.resize(num_data_);
	ordered_hessians_.resize(num_data_);

}

}