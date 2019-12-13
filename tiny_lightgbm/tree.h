#pragma once

#include<vector>

namespace Tiny_LightGBM {


class Tree {
public:

	explicit Tree(int max_leaves) ;

	inline void Split(int leaf, int feature, int real_feature,
		double left_value, double right_value, int left_cnt, int right_cnt, float gain);

	int Split(int leaf, int feature, int real_feature, int threshold_bin,
		double threshold_double, double left_value, double right_value,
		int left_cnt, int right_cnt, float gain, bool default_left);

	inline int leaf_depth(int leaf_idx) const { return leaf_depth_[leaf_idx]; }

	inline int num_leaves() const { return num_leaves_; }

	inline void Shrinkage(double rate) {

		for (int i = 0; i < num_leaves_; ++i) {
			leaf_value_[i] *= rate;
		}
		shrinkage_ *= rate;
	}

	inline double LeafOutput(int leaf) const { return leaf_value_[leaf]; }
	inline void AsConstantTree(double val) {
		num_leaves_ = 1;
		shrinkage_ = 1.0f;
		leaf_value_[0] = val;
	}

	inline int PredictLeafIndex(const double* feature_values) const;
	inline int GetLeaf(const double* feature_values) const;


	inline int NumericalDecision(double fval, int node) const {
		


		if (std::isnan(fval)) {
			
			fval = 0.0f;
			
		}
		
		if (fval <= threshold_[node]) {
			return left_child_[node];
		}
		else {
			return right_child_[node];
		}
	}


	inline double Predict(const double* feature_values) const;



	




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

inline void Tree::Split(int leaf, int feature, int real_feature,
						double left_value, double right_value, int left_cnt, int right_cnt, float gain) {

	int new_node_idx = num_leaves_ - 1;

	int parent = leaf_parent_[leaf];

	if (parent >= 0) {

	}

	split_feature_inner_[new_node_idx] = feature;
	split_feature_[new_node_idx] = real_feature;

	split_gain_[new_node_idx] = gain;
	// add two new leaves
	left_child_[new_node_idx] = ~leaf;
	right_child_[new_node_idx] = ~num_leaves_;
	// update new leaves
	leaf_parent_[leaf] = new_node_idx;
	leaf_parent_[num_leaves_] = new_node_idx;

	internal_value_[new_node_idx] = leaf_value_[leaf];
	internal_count_[new_node_idx] = left_cnt + right_cnt;
	leaf_value_[leaf] = std::isnan(left_value) ? 0.0f : left_value;
	leaf_count_[leaf] = left_cnt;
	leaf_value_[num_leaves_] = std::isnan(right_value) ? 0.0f : right_value;
	leaf_count_[num_leaves_] = right_cnt;
	// update leaf depth
	leaf_depth_[num_leaves_] = leaf_depth_[leaf] + 1;
	leaf_depth_[leaf]++;
}


inline int Tree::PredictLeafIndex(const double* feature_values) const {
	if (num_leaves_ > 1) {
		int leaf = GetLeaf(feature_values);
		return leaf;
	}
	else {
		return 0;
	}
}
inline int Tree::GetLeaf(const double* feature_values) const {
	int node = 0;
	
	
	while (node >= 0) {
		node = NumericalDecision(feature_values[split_feature_[node]], node);
	}
	
	return ~node;
}


inline double Tree::Predict(const double* feature_values) const {
	if (num_leaves_ > 1) {
		int leaf = GetLeaf(feature_values);
		return LeafOutput(leaf);
	}
	else {
		return leaf_value_[0];
	}
}




}
