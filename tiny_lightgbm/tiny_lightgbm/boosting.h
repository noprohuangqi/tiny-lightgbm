#pragma once

#include "define.h"
#include "objective_function.h"
#include "metric.h"
#include "tree_learner.h"
#include "score_updater.hpp"
#include "tree.h"

namespace Tiny_LightGBM {

class Boosting {

public:
	static Boosting* CreateBoosting();

	virtual void Init(const Dataset* train_data ,
		              const ObjectiveFunction* objective_function,
					 const std::vector<const Metric*>& training_metrics) = 0;
	virtual bool TrainOneIter(const float* gradients, const float* hessians) = 0;
	
	virtual const double* GetTrainingScore(int* out_len) = 0;

	virtual void InitPredict()=0;

	virtual int MaxFeatureIdx() const =0;
	virtual void Predict(
		const double* features, double* output) const = 0;

	virtual void PredictRaw(const double* features, double* output) const =0;

private:

};


class GBDT :public Boosting {

public:

	void Init(const Dataset* train_data,
		const ObjectiveFunction* objective_function,
		const std::vector<const Metric*>& training_metrics) override;



	virtual bool TrainOneIter(const float* gradients, const float* hessians) override;

	double BoostFromAverage(int class_id, bool update_scorer);

	void Boosting();

	const double* GetTrainingScore(int* out_len) override;

	virtual void UpdateScore(const Tree* tree, const int cur_tree_id);
	inline void InitPredict() override {
		num_iteration_for_pred_ = static_cast<int>(models_.size()) / num_tree_per_iteration_;
	}

	inline int MaxFeatureIdx() const override { return max_feature_idx_; }

	void Predict(const double* features, double* output) const override;

	void PredictRaw(const double* features, double* output) const override;

private:

	const Dataset* train_data_;

	int iter_;

	const ObjectiveFunction* objective_function_;

	std::unique_ptr<TreeLearner> tree_learner_;

	std::vector<const Metric*> training_metrics_;
	int num_tree_per_iteration_ = 1;

	bool is_constant_hessian_;

	std::unique_ptr<ScoreUpdater> train_score_updater_;

	int num_data_;

	std::vector<float> gradients_;
	std::vector<float> hessians_;

	int bag_data_cnt_;
	std::vector<int> bag_data_indices_;
	std::vector<int> tmp_indices_;
	bool is_use_subset_;

	std::vector<bool> class_need_train_;

	std::vector<std::unique_ptr<Tree>> models_;

	double shrinkage_rate_;

	int num_iteration_for_pred_;

	int max_feature_idx_;

};





}


