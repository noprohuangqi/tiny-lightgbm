#include "config.h"
#include "c_api.h"
#include "dataset.h"
#include "define.h"
#include "utils.h"
#include "bin.h"
#include "boosting.h"
#include "objective_function.h"
#include "metric.h"
#include "regression_metric.hpp"
#include "predictor.hpp"


#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <mutex>
#include <functional>


namespace Tiny_LightGBM {

class Booster {
public:

	//构造booster的几个工作，在代码解释.md里有详细讲解
	Booster(const Dataset* train_data) {

		train_data_ = train_data;

		boosting_.reset(Boosting::CreateBoosting());

		CreateObjectiveAndMetrics();

		//重头戏，GBDT
		boosting_->Init(train_data_ ,objective_fun_.get(),Utils::ConstPtrInVectorWrapper<Metric>(train_metric_) );


	}


	void CreateObjectiveAndMetrics() {

		//目标函数，即优化方向。这里就是L2loss回归
		//目标函数会影响公式计算，即一二阶导等等
		objective_fun_.reset(ObjectiveFunction::CreateObjectiveFunction());
		objective_fun_->Init(train_data_->metadata(),train_data_->num_data());

		//评价指标，不是很重要。暂时忽略
		train_metric_.clear();
		auto metric = std::unique_ptr<Metric>(Metric::CreateMetric(1));
		metric->Init(train_data_->metadata(), train_data_->num_data());
		train_metric_.push_back(std::move(metric));
		train_metric_.shrink_to_fit();

	}

	int LGBM_BoosterUpdateOneIter(void* handle, int* is_finished) {

		Booster* ref_booster = reinterpret_cast<Booster*>(handle);
		if (ref_booster->TrainOneIter()) {
			*is_finished = 1;
		}
		else {
			*is_finished = 0;
		}

	}

	bool TrainOneIter() {

		return boosting_->TrainOneIter(nullptr, nullptr);
	}

	void Predict( int nrow,
				std::function<std::vector<std::pair<int, double>>(int row_idx)> get_row_fun,
				double* out_result, int* out_len) {

		/*
		正常预测就可以了，以下都忽略
		bool is_predict_leaf = false;
		bool is_raw_score = false;
		bool predict_contrib = false;
		*/

		Predictor predictor(boosting_.get());
		//回归
		int num_pred_in_one_row = 1;

		auto pred_fun = predictor.GetPredictFunction();

		for (int i = 0; i < nrow; ++i) {
			
			auto one_row = get_row_fun(i);
			auto pred_wrt_ptr = out_result + static_cast<size_t>(num_pred_in_one_row) * i;
			pred_fun(one_row, pred_wrt_ptr);
			
		}
		*out_len = num_pred_in_one_row * nrow;
	}


private:
	const Dataset* train_data_;
	std::unique_ptr<Boosting> boosting_;

	std::unique_ptr<ObjectiveFunction> objective_fun_;

	std::vector<std::unique_ptr<Metric>> train_metric_;
};

}










using namespace Tiny_LightGBM;



std::function<std::vector<double>(int row_idx)> RowFunctionFromDenseMatric(const void* data ,
															int num_row,
															int num_col) {


	//reinterpret_cast 在cpython用的比较多，实际上是从完整结构体访问间接成员。
	//具体参考 https://www.zhihu.com/question/302752247
	const float* data_ptr = reinterpret_cast<const float*>(data);


	// 返回lambda 函数
	return [=](int row_idx) {

		std::vector<double> ret(num_col);
		auto tmp_ptr = data_ptr + static_cast<int>(num_col)*row_idx;

		for (int i = 0; i < num_col; ++i) {
			ret[i] = static_cast<double>(*(tmp_ptr + i));
		}
		return ret;
	};
}

//实际上构造dataset
Dataset* ConstructFromSampleData(double** sample_values,
								int** sample_indices,
								int num_col,
								const int* num_per_col,
								int num_row) {
	//具体参考BinMapper类的解释
	std::vector<std::unique_ptr<BinMapper>> bin_mappers(num_col);

	//针对每个原始的feature，构造bin。即抛弃原始的数据分布，进行装桶。
	//该操作实际上xgboost之前就有。可以大幅加速训练过程。
	for (int i = 0; i < num_col; ++i) {

		bin_mappers[i].reset(new BinMapper());
		bin_mappers[i]->FindBin(sample_values[i] , num_per_col[i] , num_row);

	}

	//有了每个feature的一个BinMapper之后，就可以针对整个dataset构造了
	auto dataset = std::unique_ptr<Dataset>(new Dataset(num_row));
	dataset->Construct(bin_mappers, sample_indices, num_per_col, num_row);


}

//第一个函数入口，创建dataset
//out实际上返回dataset指针， 传入是 ctypes.byref(dataset) dataset = ctypes.c_void_p()
//因此out是void** 类型
int LGBM_DatasetCreateFromMat(const void* data,
								const void* label,
								int num_row,
								int num_col,
								void** out) {
	//使用全部默认参数
	//tiny-lightgbm的实现实际上很大程度省略了参数
	//即不是所有的参数都是config中，具体会指出
	Config config;

	//通过这个函数，获取数据集的一行数据。即一个sample
	std::function<std::vector<double>(int row_idx)> get_row_fun = RowFunctionFromDenseMatric(data ,num_row,num_col );

	std::vector<std::vector<double>> sample_values(num_col);
	std::vector<std::vector<int>> sample_idx(num_col);

	for (int i = 0; i < num_row; ++i) {

		auto row = get_row_fun(static_cast<int>(i));

		//注意这里进入的元素排除了 0 ，或者说接近0的数据。
		//const double kZeroThreshold = 1e-35f;
		for (int k = 0; k < row.size(); ++k) {
			if (std::fabs(row[k]) > kZeroThreshold || std::isnan(row[k])) {

				sample_values[k].emplace_back(row[k]);
				sample_idx[k].emplace_back(static_cast<int>(i));
			}
		}
	}

	//构造dataset类，具体dataset架构参考代码解释.md
	std::unique_ptr<Dataset> ret;
	ret.reset(
			ConstructFromSampleData(
				Utils::Vector2Ptr<double>(sample_values).data(),
				Utils::Vector2Ptr<int>(sample_idx).data(),
				static_cast<int>(num_col),
				Utils::VectorSize<double>(sample_values).data(),
				static_cast<int>(num_row)
		    )
	);

	int start_row = 0;
	for (int i = 0; i < num_row; ++i) {
		auto onw_row = get_row_fun(i);
		//构造好了dataset类，就要一行一行的填充数据进去了
		ret->PushOneRow(start_row + i, onw_row);
	}


	//以上操作只填充了data，还没有填充label
	bool is_success = false;
	is_success = ret->SetFloatField(reinterpret_cast<const float*>(label));

	*out = ret.release();

	//正常返回
	return 0;
}

int LGBM_BoosterCreate(const void* train_data ,
					    void** out) {
	//转换一下dataset类
	const Dataset* p_train_data = reinterpret_cast<const Dataset*>(train_data);

	//创建核心类，booster
	auto ret = std::unique_ptr<Booster>(new Booster(p_train_data));
	*out = ret.release();
}

int LGBM_BoosterPredictForMat(void* model,
									const void* data,
									int nrow,
									int ncol,
						
									int* out_len,
									double* out_result) {
	Booster* ref_booster = reinterpret_cast<Booster*>(model);
	auto get_row_fun = RowPairFunctionFromDenseMatric(data, nrow, ncol);

	ref_booster->Predict( nrow, get_row_fun,out_result, out_len);

}

std::function<std::vector<std::pair<int, double>>(int row_idx)>
RowPairFunctionFromDenseMatric(const void* data, int num_row, int num_col) {
	auto inner_function = RowFunctionFromDenseMatric(data, num_row, num_col);
	if (inner_function != nullptr) {
		return [inner_function](int row_idx) {
			auto raw_values = inner_function(row_idx);
			std::vector<std::pair<int, double>> ret;
			for (int i = 0; i < static_cast<int>(raw_values.size()); ++i) {
				if (std::fabs(raw_values[i]) > kZeroThreshold || std::isnan(raw_values[i])) {
					ret.emplace_back(i, raw_values[i]);
				}
			}
			return ret;
		};
	}
	return nullptr;
}

