#pragma once

#include "dataset.h"


namespace Tiny_LightGBM {

class TreeLearner {
	
public:
	static TreeLearner* CreateTreeLearner();

	virtual void Init(const Dataset* train_data) = 0;

};


class SerialTreeLearner :public TreeLearner {
public:
	void Init(const Dataset* train_data) override;

protected:

	const Dataset* train_data_;
	int num_data_;
	int num_features_;



};



}