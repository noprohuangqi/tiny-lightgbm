#pragma once

#include "dataset.h"
#include <functional>



namespace Tiny_LightGBM {

class ObjectiveFunction {

public:

	virtual ~ObjectiveFunction() {}

	static ObjectiveFunction* CreateObjectiveFunction();

	virtual void Init(const Metadata& metadata, int num_data) = 0;



};


class RegressionL2loss :public ObjectiveFunction {

public:
	void Init(const Metadata& metadata, int num_data) override {};


//提供给派生类
protected:
	int num_data_;
	const float* label_;

};








}
