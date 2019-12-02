#pragma once

#include "define.h"
#include "objective_function.h"
#include "metric.h"
#include "tree_learner.h"
#include "score_updater.hpp"


namespace Tiny_LightGBM {

class Boosting {

public:
	static Boosting* CreateBoosting();

	virtual void Init(const Dataset* train_data ,
		              const ObjectiveFunction* objective_function,
					 const std::vector<const Metric*>& training_metrics) = 0;
	virtual bool TrainOneIter(const float* gradients, const float* hessians) = 0;
private:

};


class GBDT :public Boosting {

public:

	void Init(const Dataset* train_data,
		const ObjectiveFunction* objective_function,
		const std::vector<const Metric*>& training_metrics) override;



	virtual bool TrainOneIter(const float* gradients, const float* hessians) override;

private:

	const Dataset* train_data_;

	int iter_;

	const ObjectiveFunction* objective_function_;

	std::unique_ptr<TreeLearner> tree_learner_;

	std::vector<const Metric*> training_metrics_;
	int num_tree_per_iteration_ = 1;

	std::unique_ptr<ScoreUpdater> train_score_updater_;

	int num_data_;

	std::vector<float> gradients_;
	std::vector<float> hessians_;

	int bag_data_cnt_;
	std::vector<int> bag_data_indices_;
	std::vector<int> tmp_indices_;
	bool is_use_subset_;

	std::vector<bool> class_need_train_;



};





}


