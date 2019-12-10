

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
	is_constant_hessian_ = true;

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


bool GBDT::TrainOneIter(const float* gradients, const float* hessians) {

	//
	// 省略了baggin的操作
	//
	// 0.0 默认就是double ， 不需要类型转换。0.0f默认就是float
	std::vector<double> init_scores(num_tree_per_iteration_, 0.0);

	for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {

		//init_scores[cur_tree_id] = BoostFromAverage(cur_tree_id, true);

	}
	Boosting();
	gradients = gradients_.data();
	hessians = hessians_.data();

	//省略bagging操作

	bool should_continue = false;
	for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id){
		const int bias = cur_tree_id * num_data_;
		std::unique_ptr<Tree> new_tree(new Tree(2));
		
		auto grad = gradients + bias;

		auto hess = hessians + bias;

		new_tree.reset(tree_learner_->)


	}


}

void GBDT::Boosting() {

	int num_score = 0;

	objective_function_->GetGradients(GetTrainingScore(&num_score), gradients_.data(), hessians_.data());

}

const double* GBDT::GetTrainingScore(int* out_len) {

	return train_score_updater_->score();


}


double GBDT::BoostFromAverage(int class_id, bool update_scorer) {

	if (models_.empty() && !train_score_updater_->has_init_score()) {

		double init_score = 0.0;
	}
	return 0.0f;
}



}