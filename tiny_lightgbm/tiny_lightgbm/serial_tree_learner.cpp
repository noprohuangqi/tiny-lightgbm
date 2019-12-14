

#include "tree_learner.h"
#include "config.h"

namespace Tiny_LightGBM {


void SerialTreeLearner::Init(const Dataset* train_data) {
	
	train_data_ = train_data;
	num_data_ = train_data_->num_data();
	//这个就是num total feature有所不同，但是在我们的简单案例下是相同的
	num_features_ = train_data_->num_features();

	//默认是31
	//超参
	int max_cache_size = Config::num_leaves;

	//重头戏
	histogram_pool_.DynamicChangeSize(train_data_,max_cache_size);

	//以下都是麻瓜初始化，为了分裂树时所必须记录的信息
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

	//每棵树最多训练出31个叶子（超参），在训练之前，先要做一些初始化的麻瓜工作
	BeforeTrain();

	//初始化tree
	auto tree = std::unique_ptr<Tree>(new Tree(Config::num_leaves));

	//leaf的index从0开始
	//这里有一个分裂的技巧，假设最开始是根节点，也是叶子，赋予index 0 ；
	//分裂一次，有了两个新的叶子，左叶子继承0，右叶子变为1。以此类推
	//最后分裂完成，叶子的index是0-30（31个叶子）
	int left_leaf = 0;
	int cur_depth = 1;
	int right_leaf = -1;
	int init_splits = 0;
	
	//开始分裂，最多分裂30次，就会有31个叶子（leaf-wise）
	for (int split = init_splits; split < Config::num_leaves; ++split) {

		//每次分裂叶子之前，检查
		if (BeforeFindBsetSplit(tree.get(), left_leaf, right_leaf)) {

			FindBestSplits();
		}
		int best_leaf = static_cast<int>(Utils::ArgMax(best_split_per_leaf_));
		const SplitInfo& best_leaf_SplitInfo = best_split_per_leaf_[best_leaf];

		if (best_leaf_SplitInfo.gain <= 0.0) {
			break;
		}

		Split(tree.get(), best_leaf, &left_leaf, &right_leaf);
		cur_depth = std::max(cur_depth, tree->leaf_depth(left_leaf));

	}
	return tree.release();
}

void SerialTreeLearner::Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf) {

	const SplitInfo& best_split_info = best_split_per_leaf_[best_leaf];
	const int inner_feature_index = train_data_->InnerFeatureIndex(best_split_info.feature);

	//root的情况下，将best_leaf = 0赋给了leftleaf
	*left_leaf = best_leaf;

	auto threshold_double = train_data_->RealThreshold(inner_feature_index, best_split_info.threshold);
	*right_leaf = tree->Split(best_leaf,
								inner_feature_index,
								best_split_info.feature,
								best_split_info.threshold,
								threshold_double,
								static_cast<double>(best_split_info.left_output),
								static_cast<double>(best_split_info.right_output),
								static_cast<int>(best_split_info.left_count),
								static_cast<int>(best_split_info.right_count),
								static_cast<float>(best_split_info.gain),
								best_split_info.default_left);
	data_partition_->Split(best_leaf, train_data_, inner_feature_index,
							&best_split_info.threshold, 1, best_split_info.default_left, *right_leaf);


	auto p_left = smaller_leaf_splits_.get();
	auto p_right = larger_leaf_splits_.get();
	if (best_split_info.left_count < best_split_info.right_count) {
		smaller_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
		larger_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);

	}
	else {
		smaller_leaf_splits_->Init(*right_leaf, data_partition_.get(), best_split_info.right_sum_gradient, best_split_info.right_sum_hessian);
		larger_leaf_splits_->Init(*left_leaf, data_partition_.get(), best_split_info.left_sum_gradient, best_split_info.left_sum_hessian);
		p_right = smaller_leaf_splits_.get();
		p_left = larger_leaf_splits_.get();
	}
	p_left->SetValueConstraint(best_split_info.min_constraint, best_split_info.max_constraint);
	p_right->SetValueConstraint(best_split_info.min_constraint, best_split_info.max_constraint);


}

void SerialTreeLearner::FindBestSplits() {

	std::vector<int> is_feature_used(num_features_, 0);

	for (int feature_index = 0;feature_index < num_features_; ++feature_index) {
		
		//if (parent_leaf_histogram_array_ != nullptr
		//	&& !parent_leaf_histogram_array_[feature_index].is_splittable()) {

		//	smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
		//	continue;
		//}

		//is_feature_used_全选
		is_feature_used[feature_index] = 1;
	}
	//情况一：只有根节点。use_subtract=false
	//情况二：use_subtract=true
	bool use_subtract = parent_leaf_histogram_array_ != nullptr;
	
	
	ConstructHistograms(is_feature_used, use_subtract);
	FindBestSplitsFromHistograms(is_feature_used, use_subtract);
}
void SerialTreeLearner::FindBestSplitsFromHistograms(const std::vector<int>& is_feature_used, bool use_subtract) {

	std::vector<SplitInfo> smaller_best(1);
	std::vector<SplitInfo> larger_best(1);
	//原始的feature，即不是用group
	for (int feature_index = 0; feature_index < num_features_; ++feature_index) {


		SplitInfo smaller_split;

		//fixhistogram是假设某个feature的default bin不在0上
		//那么真实的value 0 对应的bin没有数据在上面，都去了bin 0
		//这个函数就可以还原数据到正确的bin上面去

		//再有重点：
		//之前构造histogram的时候，是按照group来构造的。
		//但是这里分裂的时候，是按照原始feature来分裂的。
		//所以fix是在原始feature上fix
		//即之前构造的bin0，bin1，bin2，....，bin10
		//其中可能bin 0 是默认，bin 1-5是feature1 ， 剩下的是feature2
		//这个时候feature1的default bin是在bin3，那么就需要fix，因为bin 3现在没有任何数据
		//fix的时候，是需要把bin 0- bin10的全部数据，减去bin0 ，bin1，bin2，bin4，bin5.
		//这样操作之后，对于feature1来说，其实他已经囊括了所有的数据

		//smaller_leaf_histogram_array_这个参数就代表一个叶子了
		//smaller_leaf_histogram_array_[feature_index]就换到feature上了
		train_data_->FixHistogram(feature_index,
									smaller_leaf_splits_->sum_gradients(), smaller_leaf_splits_->sum_hessians(),
									smaller_leaf_splits_->num_data_in_leaf(),
									smaller_leaf_histogram_array_[feature_index].RawData());
		int real_fidx = train_data_->RealFeatureIndex(feature_index);
		//寻找最佳分裂点
		smaller_leaf_histogram_array_[feature_index].FindBestThreshold(smaller_leaf_splits_->sum_gradients(),
												smaller_leaf_splits_->sum_hessians(),
												smaller_leaf_splits_->num_data_in_leaf(),
												smaller_leaf_splits_->min_constraint(),
												smaller_leaf_splits_->max_constraint(),
												&smaller_split);

		smaller_split.feature = real_fidx;
		if (smaller_split > smaller_best[0]) {
			smaller_best[0] = smaller_split;
		}
		if (larger_leaf_splits_ == nullptr || larger_leaf_splits_->LeafIndex() < 0) { continue; }

		if (use_subtract) {
			larger_leaf_histogram_array_[feature_index].Subtract(smaller_leaf_histogram_array_[feature_index]);
		}
		else {
			train_data_->FixHistogram(feature_index, larger_leaf_splits_->sum_gradients(), larger_leaf_splits_->sum_hessians(),
				larger_leaf_splits_->num_data_in_leaf(),
				larger_leaf_histogram_array_[feature_index].RawData());
		}
		SplitInfo larger_split;
		// find best threshold for larger child
		larger_leaf_histogram_array_[feature_index].FindBestThreshold(
			larger_leaf_splits_->sum_gradients(),
			larger_leaf_splits_->sum_hessians(),
			larger_leaf_splits_->num_data_in_leaf(),
			larger_leaf_splits_->min_constraint(),
			larger_leaf_splits_->max_constraint(),
			&larger_split);
		larger_split.feature = real_fidx;
		if (larger_split > larger_best[0]) {
			larger_best[0] = larger_split;
		}


	}

	auto smaller_best_idx = Utils::ArgMax(smaller_best);
	int leaf = smaller_leaf_splits_->LeafIndex();
	best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];

	//情况一：只有根节点。这段代码不会执行
	if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
		leaf = larger_leaf_splits_->LeafIndex();
		auto larger_best_idx = Utils::ArgMax(larger_best);
		best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
	}

}


void SerialTreeLearner::ConstructHistograms(const std::vector<int>& is_feature_used, bool use_subtract) {

	//情况一：为根节点执行,跟节点就是smaller_leaf_histogram_array_
	//[0]代表我们从feature0开始，构造整个histogram
	//每次只会构造small这边的histogram，larger只需要做个减法
	HistogramBinEntry* ptr_smaller_leaf_hist_data = smaller_leaf_histogram_array_[0].RawData() - 1;
	train_data_->ConstructHistograms(is_feature_used,
								smaller_leaf_splits_->data_indices(), smaller_leaf_splits_->num_data_in_leaf(),
								smaller_leaf_splits_->LeafIndex(),
								ordered_bins_, gradients_, hessians_,
								ordered_gradients_.data(), ordered_hessians_.data(), is_constant_hessian_,
								ptr_smaller_leaf_hist_data);

	//永远不会执行，即不需要复杂的为larger_leaf构造histogram
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

	//检查叶子还够不够分，如果叶子上的数据数量已经很少了，就算了
	if (num_data_in_left_child < static_cast<int>(Config::min_data_in_leaf * 2)
		&& num_data_in_right_child < static_cast<int>(Config::min_data_in_leaf * 2)) {

		best_split_per_leaf_[left_leaf].gain = -std::numeric_limits<float>::infinity();
		if (right_leaf > 0) {
			best_split_per_leaf_[right_leaf].gain = -std::numeric_limits<float>::infinity();
		}
		return false;

	}

	if (right_leaf < 0) {
		//最开始只有根节点，显然要分裂，就把根节点给了smaller_leaf_histogram_array_
		histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
		larger_leaf_histogram_array_ = nullptr;
	}
	else if (num_data_in_left_child < num_data_in_right_child) {

		// 通过swap操作，实现了左右树的交换
		//这里主要就是左右叶子要有个顺序，即左叶子数据要多一些。默认根节点也是左叶子（顺理成章）
		//自然左叶子就连到了larger_leaf_histogram_array_
		if (histogram_pool_.Get(left_leaf , &larger_leaf_histogram_array_)) {
			parent_leaf_histogram_array_ = larger_leaf_histogram_array_;
		}
		histogram_pool_.Move(left_leaf, right_leaf);
		histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
	}
	else {
		if (histogram_pool_.Get(left_leaf, &larger_leaf_histogram_array_)) { 
			parent_leaf_histogram_array_ = larger_leaf_histogram_array_; 
		}
		histogram_pool_.Get(right_leaf, &smaller_leaf_histogram_array_);
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