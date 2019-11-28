#include "boosting.h"



namespace Tiny_LightGBM {

Boosting* Boosting::CreateBoosting() {

	return new GBDT();


}


}