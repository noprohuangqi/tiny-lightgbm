

#include "tree_learner.h"


namespace Tiny_LightGBM {


TreeLearner* TreeLearner::CreateTreeLearner() {
	
	return new SerialTreeLearner();
}






}