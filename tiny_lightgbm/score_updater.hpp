#pragma once


namespace Tiny_LightGBM {

class ScoreUpdater {

public:
	ScoreUpdater(const Dataset* data, int num_tree_per_iteration) :data_(data) {
		num_data_ = data->num_data();
		int total_size = static_cast<int>(num_data_) * num_tree_per_iteration;

		score_.resize(total_size);

		for (int i = 0; i < total_size; ++i) {
			score_[i] = 0.0f;
		}
		has_init_score_ = false;

		//没有使用init
		const double* init_score = nullptr;
		//省略了部分内容
	}

	inline bool has_init_score() const { return has_init_score_; }

	inline const double* score() const { return score_.data(); }

	inline void AddScore(const TreeLearner* tree_learner, const Tree* tree, int cur_tree_id) {
		const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
		tree_learner->AddPredictionToScore(tree, score_.data() + offset);
	}

	inline void AddScore(double val, int cur_tree_id) {
		const size_t offset = static_cast<size_t>(num_data_) * cur_tree_id;
		for (int i = 0; i < num_data_; ++i) {
			score_[offset + i] += val;
		}
	}

private:
	int num_data_;
	const Dataset* data_;
	std::vector<double> score_;
	bool has_init_score_;


};

}
