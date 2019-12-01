

#include "gbdt.h"
#include "boosting.h"
#include "tree_learner.h"


namespace Tiny_LightGBM {

void GBDT::Init(const Dataset* train_data,
	const ObjectiveFunction* objective_function,
	const std::vector<const Metric*>& training_metrics) {

	train_data_ = train_data;
	iter_ = 0;

	objective_function_ = objective_function;
	num_tree_per_iteration_ = 1;

	tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner());

	tree_learner_->Init(train_data_);

	training_metrics_.clear();
	for (const auto& metric : training_metrics) {
		training_metrics_.push_back(metric);
	}
	training_metrics_.shrink_to_fit();

	train_score_updater_.reset(new ScoreUpdater(train_data_, num_tree_per_iteration_));

	num_data_ = train_data_->num_data();

	int total_size = static_cast<int>(num_data_)*num_tree_per_iteration_;
	gradients_.resize(total_size);
	hessians_.resize(total_size);

	bag_data_cnt_ = num_data_;
	bag_data_indices_.clear();
	tmp_indices_.clear();
	is_use_subset_ = false;

	class_need_train_ = std::vector<bool>(num_tree_per_iteration_, true);


}



}