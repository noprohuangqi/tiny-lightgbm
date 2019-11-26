#pragma once

#include<cstdint>
#include<cstring>
#include "define.h"





TINY_LIGHTGBM_C_EXPORT int LGBM_DatasetCreateFromMat(const void* data,
													const void* label,
													int row,
													int col,
													void* out);


