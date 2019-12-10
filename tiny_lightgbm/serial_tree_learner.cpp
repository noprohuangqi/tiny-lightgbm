

#include "tree_learner.h"
#include "config.h"

namespace Tiny_LightGBM {


void SerialTreeLearner::Init(const Dataset* train_data) {
	
	train_data_ = train_data;
	num_data_ = train_data_->num_data();
	num_features_ = train_data_->num_features();

	//默认是31
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

Tree* SerialTreeLearner::Train(const float* gradients, const float* hessians, bool is_constant_hessian) {

	gradients_ = gradients;
	hessians_ = hessians;
	is_constant_hessian_ = is_constant_hessian;

	BeforeTrain();

	auto tree = std::unique_ptr<Tree>(new Tree(Config::num_leaves));

	int left_leaf = 0;
	int cur_depth = 1;
	int right_leaf = -1;
	int init_splits = 0;
	

	for (int split = init_splits; split < Config::num_leaves; ++split) {

		if (BeforeFindBsetSplit(tree.get(), left_leaf, right_leaf)) {

			
		}


	}




}

void SerialTreeLearner::FindBestSplits() {

	std::vector<int> is_feature_used(num_features_, 0);

	for (int feature_index = 0;feature_index < num_features_; ++feature_index) {
		//is_feature_used_全选
		if (parent_leaf_histogram_array_ != nullptr
			&& !parent_leaf_histogram_array_[feature_index].is_splittable()) {

			smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
			continue;
		}

		is_feature_used[feature_index] = 1;
	}
	bool use_subtract = parent_leaf_histogram_array_ != nullptr;
	
	ConstructHistograms(is_feature_used, use_subtract);
	FindBestSplitsFromHistograms(is_feature_used, use_subtract);



}
void SerialTreeLearner::FindBestSplitsFromHistograms(const std::vector<int>& is_feature_used, bool use_subtract) {

	std::vector<SplitInfo> smaller_best(1);
	std::vector<SplitInfo> larger_best(1);




}


void SerialTreeLearner::ConstructHistograms(const std::vector<int>& is_feature_used, bool use_subtract) {

	HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
	train_data_->ConstructHistograms(is_feature_used,
								smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
								smaller_leaf_splits_->LeafIndex(),
								ordered_bins_, gradients_, hessians_,
								ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
								ptr_smaller_leaf_hist_data);

	if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
		// construct larger leaf
		HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
		train_data_->ConstructHistograms(is_feature_used,
			smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
			smaller_leaf_splits_->LeafIndex(),
			ordered_bins_, gradients_, hessians_,
			ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
			ptr_smaller_leaf_hist_data);

		if (larger_leaf_histogram_array_ != nullptr && !use_subtract) {
			// construct larger leaf
			HistogramBinEntry* ptr_larger_leaf_hist_data = larger_leaf_histogram_array_[0].RawData() - 1;
			train_data_->ConstructHistograms(is_feature_used,
				larger_leaf_splits_->data_indices(), larger_leaf_splits_->num_data_in_leaf(),
				larger_leaf_splits_->LeafIndex(),
				ordered_bins_, gradients_, hessians_,
				ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
				ptr_larger_leaf_hist_data);
		}
	}
}


bool SerialTreeLearner::BeforeFindBsetSplit(const Tree* tree, int left_leaf, int right_leaf) {

	//max_depth没有限制

	int num_data_in_left_child = GetGlobalDataCountInLeaf(left_leaf);
	int num_data_in_right_child = GetGlobalDataCountInLeaf(right_leaf);

	if (num_data_in_left_child < static_cast<int>(Config::min_data_in_leaf * 2)
		&& num_data_in_right_child < static_cast<int>(Config::min_data_in_leaf * 2)) {

		best_split_per_leaf_[left_leaf].gain = -std::numeric_limits<float>::infinity();
		if (right_leaf > 0) {
			best_split_per_leaf_[right_leaf].gain = -std::numeric_limits<float>::infinity();
		}
		return false;

	}

	if (right_leaf < 0) {
		histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
		larger_leaf_histogram_array_ = nullptr;
	}
	else if (num_data_in_left_child < num_data_in_right_child) {
		if (histogram_pool_.Get(left_leaf)) {

		}


	}
	return true;



}

void SerialTreeLearner::BeforeTrain() {

	histogram_pool_.ResetMap();

	for (int i = 0; i < num_features_; ++i) {
		is_feature_used_[i] = 1;

	}
	data_partition_->Init();


	for (int i = 0; i < Config::num_leaves; ++i) {
		best_split_per_leaf_[i].Reset();
	}

	smaller_leaf_splits_->Init(0,data_partition_.get(), gradients_, hessians_);
	larger_leaf_splits_->Init();
}



}