#pragma once

#include "boosting.h"

#define PredictFunction std::function<void(const std::vector<std::pair<int, double>>&, double* output)>
namespace Tiny_LightGBM {




class Predictor {

public:
	Predictor(Boosting* boosting) {

		boosting->InitPredict();

		boosting_ = boosting;

		//»Ø¹é
		num_pred_one_row_ = 1;
		num_feature_ = boosting_->MaxFeatureIdx() + 1;

		predict_buf_ = std::vector<std::vector<double>>(1, std::vector<double>(num_feature_, 0.0f));

		

		predict_fun_ = [=](const std::vector<std::pair<int, double>>& features, double* output) {

			CopyToPredictBuffer(predict_buf_[0].data(), features);
			boosting_->predict(predict_buf_[0].data(), output);
			ClearPredictBuffer(predict_buf_[0].data(), predict_buf_[0].size(), features);
			
		};
	}

	void ClearPredictBuffer(double* pred_buf, size_t buf_size, const std::vector<std::pair<int, double>>& features) {
		if (features.size() > static_cast<size_t>(buf_size / 2)) {
			std::memset(pred_buf, 0, sizeof(double)*(buf_size));
		}
		else {
			int loop_size = static_cast<int>(features.size());
			for (int i = 0; i < loop_size; ++i) {
				if (features[i].first < num_feature_) {
					pred_buf[features[i].first] = 0.0f;
				}
			}
		}
	}

	inline const PredictFunction& GetPredictFunction() const {
		return predict_fun_;
	}

private:
	const Boosting* boosting_;
	int num_pred_one_row_;
	int num_feature_;

	std::vector<std::vector<double>> predict_buf_;

	PredictFunction predict_fun_;

	void CopyToPredictBuffer(double* pred_buf, const std::vector<std::pair<int, double>>& features) {
		int loop_size = static_cast<int>(features.size());
		for (int i = 0; i < loop_size; ++i) {
			if (features[i].first < num_feature_) {
				pred_buf[features[i].first] = features[i].second;
			}
		}
	}
};



}
