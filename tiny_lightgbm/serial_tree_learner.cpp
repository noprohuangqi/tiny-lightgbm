

#include "tree_learner.h"
#include "config.h"

namespace Tiny_LightGBM {


void SerialTreeLearner::Init(const Dataset* train_data) {
	
	train_data_ = train_data;
	num_data_ = train_data_->num_data();
	num_features_ = train_data_->num_features();

	//д╛хойг31
	int max_cache_size = Config::num_leaves;

	histogram_pool_.DynamicChangeSize(train_data_,max_cache_size);



}

}