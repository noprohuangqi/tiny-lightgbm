# tiny-lightgbm
一个简单的lightgbm的实现，主要是为了学习c++和机器学习算法。

A simple implementation of lightgbm , only for study source code(cpython , c++ , python) and gbdt.

目前所有的注释是中文，未来可能会更新英文注释。

And now all note was written by Chinese , English note may be completed in the future work.

## 功能缺失

1. 没有对缺失值，异常值（nan）的处理。
2. 目前只支持基本的回归任务。feature支持float32 ， label支持float32
3. 暂时只支持feature全部是连续数值类型（numerica），catefory暂时不支持。
4. 暂时不能处理缺失值。
5. 没有加入EFB ， 即不考虑比较稀疏的特征的处理。
6. 