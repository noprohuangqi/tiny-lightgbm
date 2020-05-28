EFB是用来加速训练的，将数量较多的稀疏特征变为数量较少的稠密特征。

在Config中，通过 enable_bundle 参数控制是否使用EFB。默认使用。

EFB反映在算法里，实际上是返回一个 std::vector < std::vector<int> >

即代表每一个feature group由哪些feature index组成。



具体：

首先对所有feature里非零元素的数量进行排序。



