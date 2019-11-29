#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>


namespace Tiny_LightGBM{

//保存所有参数，即在python接口中不提供参数选项。
//所有的修改都在配置文件config.h中进行
struct Config {

public:

	static const int max_bin = 255;

	static const int min_data_in_bin = 3;

	

	static const int num_leaves = 31;

};


}
