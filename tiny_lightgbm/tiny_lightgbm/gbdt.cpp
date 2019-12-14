

#include "gbdt.h"
#include "boosting.h"
#include "tree_learner.h"


namespace Tiny_LightGBM {

void GBDT::Init(const Dataset* train_data,
	const ObjectiveFunction* objective_function,
	const std::vector<const Metric*>& training_metrics) {

	//麻瓜初始化
	train_data_ = train_data;
	iter_ = 0;
	objective_function_ = objective_function;
	//这个参数是代表每一个epoch里面，会训练多少棵树
	//lightgbm处理多分类（不是多标签）问题时，假设有5类
	//那么每个epoch就会构造5棵树，每棵树对应一个类别，实现二分类
	//最后来个softmax可以得到概率之类的（分类也是回归树）
	num_tree_per_iteration_ = 1;
	//显然L2loss下面，二阶导常数
	is_constant_hessian_ = true;

	//重头戏
	tree_learner_ = std::unique_ptr<TreeLearner>(TreeLearner::CreateTreeLearner());
	tree_learner_->Init(train_data_);


	//以下都是麻瓜初始化
	training_metrics_.clear();
	for (const auto& metric : training_metrics) {
		training_metrics_.push_back(metric);
	}
	training_metrics_.shrink_to_fit();

	//最开始所有的data score都是0
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

	//默认不进行shrinkage
	shrinkage_rate_ = 1.0f;

	//这个是所有原始feature，并不是EFB之后的
	max_feature_idx_ = 0;
	max_feature_idx_ = train_data_->num_total_features() - 1;


}

void GBDT::UpdateScore(const Tree* tree, const int cur_tree_id) {

	//更新
	train_score_updater_->AddScore(tree_learner_.get(), tree, cur_tree_id);

}

//开始训练，核心函数
bool GBDT::TrainOneIter(const float* gradients, const float* hessians) {

	// 0.0 默认就是double ，不需要类型转换。0.0f默认就是float
	//其实initscore可以是label的平均值，加速训练，这里忽略
	std::vector<double> init_scores(num_tree_per_iteration_, 0.0);

	//计算当前的一二阶导
	//显然每个epoch的一二阶导都不一样，因为可以理解为
	//每个epoch的初始值都会离真实label越来越近
	Boosting();
	gradients = gradients_.data();
	hessians = hessians_.data();

	//省略bagging操作，所有的baggin都没有做。默认使用全部数据，全部feature

	bool should_continue = false;
	for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id){
		const int bias = cur_tree_id * num_data_;

		//每个epoch都会有num_tree_per_iteration_这么多个树
		std::unique_ptr<Tree> new_tree(new Tree(2));
		
		auto grad = gradients + bias;
		auto hess = hessians + bias;
		//核心函数，开始训练一棵树
		new_tree.reset(tree_learner_->Train(grad, hess, is_constant_hessian_));

		//训练完毕，树有所分裂，那么添加到模型，并更新数据记录
		if (new_tree->num_leaves() > 1) {
			should_continue = true;
			new_tree->Shrinkage(shrinkage_rate_);
			//这个score更新之后，就会用到下一次的计算一二阶导
			UpdateScore(new_tree.get(), cur_tree_id);
		}
		//反之，不更新树，并且停止所有后续的训练。给个分值为0。结束
		else {
			if (models_.size() < static_cast<size_t>(num_tree_per_iteration_)) {
				double output = 0.0;
				output = init_scores[cur_tree_id];
				new_tree->AsConstantTree(output);
				train_score_updater_->AddScore(output, cur_tree_id);
			}

		}

		models_.push_back(std::move(new_tree));
	}

	if (!should_continue) {
		if (models_.size() > static_cast<size_t>(num_tree_per_iteration_)) {
			for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
				//最后一棵树没有用，踢出
				models_.pop_back();
			}
		}
		return true;
	}

	++iter_;
	return false;


}

void GBDT::Boosting() {

	int num_score = 0;

	//计算一二阶导
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


void GBDT::Predict(const double* features, double* output) const {

	PredictRaw(features, output);
	
}

void GBDT::PredictRaw(const double* features, double* output) const {

	std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);

	for (int i = 0; i < num_iteration_for_pred_; ++i) {
		// predict all the trees for one iteration
		for (int k = 0; k < num_tree_per_iteration_; ++k) {
			output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
		}
		
	}
}


}