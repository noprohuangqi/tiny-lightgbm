#pragma once

#include<vector>

namespace Tiny_LightGBM {


class Tree {
public:

	explicit Tree(int max_leaves) ;

private:
	int max_leaves_;
	int num_leaves_;

	std::vector<int> left_child_;

	std::vector<int> right_child_;
	std::vector<int> split_feature_inner_;
	/*! \brief A non-leaf node's split feature, the original index */
	std::vector<int> split_feature_;
	std::vector<uint32_t> threshold_in_bin_;
	/*! \brief A non-leaf node's split threshold in feature value */
	std::vector<double> threshold_;
	int num_cat_;
	std::vector<int> cat_boundaries_inner_;
	std::vector<uint32_t> cat_threshold_inner_;
	std::vector<int> cat_boundaries_;
	std::vector<uint32_t> cat_threshold_;
	/*! \brief Store the information for categorical feature handle and mising value handle. */
	std::vector<int8_t> decision_type_;
	/*! \brief A non-leaf node's split gain */
	std::vector<float> split_gain_;
	// used for leaf node
	/*! \brief The parent of leaf */
	std::vector<int> leaf_parent_;
	/*! \brief Output of leaves */
	std::vector<double> leaf_value_;
	/*! \brief DataCount of leaves */
	std::vector<int> leaf_count_;
	/*! \brief Output of non-leaf nodes */
	std::vector<double> internal_value_;
	/*! \brief DataCount of non-leaf nodes */
	std::vector<int> internal_count_;
	/*! \brief Depth for leaves */
	std::vector<int> leaf_depth_;
	double shrinkage_;
	int max_depth_;


};

}
