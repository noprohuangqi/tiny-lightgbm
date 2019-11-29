#pragma once

#include "define.h"
#include "objective_function.h"
#include "metric.h"
#include "tree_learner.h"


namespace Tiny_LightGBM {

class Boosting {

public:
	static Boosting* CreateBoosting();

	virtual void Init(const Dataset* train_data ,
		              const ObjectiveFunction* objective_function,
					 const std::vector<const Metric*>& training_metrics) = 0;

private:

};


class GBDT :public Boosting {

public:

	void Init(const Dataset* train_data,
		const ObjectiveFunction* objective_function,
		const std::vector<const Metric*>& training_metrics) override;



private:

	const Dataset* train_data_;

	int iter_;

	const ObjectiveFunction* objective_function_;

	std::unique_ptr<TreeLearner> tree_learner_;

};





}


