
#include "tree.h"

namespace Tiny_LightGBM {

Tree::Tree(int max_leaves):max_leaves_(max_leaves) {
	left_child_.resize(max_leaves_ - 1);
	right_child_.resize(max_leaves_ - 1);
	split_feature_inner_.resize(max_leaves_ - 1);
	split_feature_.resize(max_leaves_ - 1);
	threshold_in_bin_.resize(max_leaves_ - 1);
	threshold_.resize(max_leaves_ - 1);
	decision_type_.resize(max_leaves_ - 1, 0);
	split_gain_.resize(max_leaves_ - 1);
	leaf_parent_.resize(max_leaves_);
	leaf_value_.resize(max_leaves_);
	leaf_count_.resize(max_leaves_);
	internal_value_.resize(max_leaves_ - 1);
	internal_count_.resize(max_leaves_ - 1);
	leaf_depth_.resize(max_leaves_);
	// root is in the depth 0
	leaf_depth_[0] = 0;
	num_leaves_ = 1;
	leaf_value_[0] = 0.0f;
	leaf_parent_[0] = -1;
	shrinkage_ = 1.0f;
	num_cat_ = 0;
	cat_boundaries_.push_back(0);
	cat_boundaries_inner_.push_back(0);
	max_depth_ = -1;



}

int Tree::Split(int leaf, int feature, int real_feature, int threshold_bin,
					double threshold_double, double left_value, double right_value,
					int left_cnt, int right_cnt, float gain, bool default_left) {

	Split(leaf, feature, real_feature, left_value, right_value, left_cnt, right_cnt, gain);
	++num_leaves_;
	return num_leaves_ - 1;
}


}