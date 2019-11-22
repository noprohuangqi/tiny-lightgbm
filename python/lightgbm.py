## some python interface for tiny lightgbm
## 只实现最简单的功能，即二分类任务
## 尽可能在不影响主干的情况下精简代码，仅供学习使用

## author:nopro
## 2019.11.21


import ctypes
import numpy as np
import sklearn.metrics as sm
from sklearn.datasets import load_iris


## dll
_LIB


def train(params , X_train , y_train):
    '''
    参数即语义
    返回值是一个ctypes下的指针，指向model
    '''

    # 处理参数
    params = {'objective':'binary' ,
              'num_iteration': 100,
              'learning_rates': 0.1,}
    params_c = c_str(param_dict_to_str(params))

    # 新建两个指针，分别对应model和data
    model = ctypes.c_void_p()
    dataset = ctypes.c_void_p()

    # 处理数据
    X_train_c ,num_row ,num_col= reshape_to_c(X_train ,is_label = False)
    y_train_c ,_ ,_= reshape_to_c(y_train , is_label = True)

    ## 得到dataset
    _LIB.LGBM_DatasetCreateFromMat(
            X_train_c,
            y_train_c,
            ctypes.c_int(num_row),
            ctypes.c_int(num_col),
            params_c,
            ctypes.byref(dataset))


    ## 初始化model
    _LIB.LGBM_BoosterCreate(
            dataset,
            params_c,
            ctypes.byref(model))

    ## 训练model
    for i in range(100):

        _LIB.LGBM_BoosterUpdateOneIter(
                model,
                ctypes.byref(is_finished))

        # 设置早停
        if is_finished.value != 1:
            break

    return model






def predict(model , X_test):
    '''

    '''

    X_test ,num_row ,num_col= reshape_to_c(X_test ,is_label = False)

    preds = np.zeros(num_row)

    _LIB.LGBM_BoosterPredictForMat(
                model,
                X_test,
                ctypes.c_int(num_row),
                preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    return preds









def load_data():


if __name__ == "__main__":
    '''
    测试使用
    出于精简考虑，我们使用train去获得model
    使用predict获得预测
    使用eval对比结果
    '''

    # 使用iris数据集简单测试，只保留一部分数据做个二分类
    # 数据格式：float32
    X_train , y_train , X_test , y_test = load_data()

    #params 使用默认
    model = train('' , x_train , y_train)

    #预测结果
    y_predict = predict(model , X_test)

    #进行评估
    acc = sm.acc_score(y_test , y_predict)
    print(acc)

    print('done')













