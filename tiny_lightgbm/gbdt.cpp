

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

	tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner());

	tree_learner_->Init(train_data_);



}



}