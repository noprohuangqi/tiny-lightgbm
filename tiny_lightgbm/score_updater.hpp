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

private:
	int num_data_;
	const Dataset* data_;
	std::vector<double> score_;
	bool has_init_score_;


};

}
