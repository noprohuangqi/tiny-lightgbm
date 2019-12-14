
#include "objective_function.h"
#include "dataset.h"



namespace Tiny_LightGBM {

ObjectiveFunction* ObjectiveFunction::CreateObjectiveFunction() {

	return new RegressionL2loss();

}

void RegressionL2loss::Init(const Metadata& metadata, int num_data) {

	num_data_ = num_data;
	label_ = metadata.label();


}


}