import torch as th


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


# 继承Transform
class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        # 创建一个新的向量，最后一个维度变为4，其他维度和tensor原来的维度不变
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        # -1表示在最后一个维度，tensor是索引，1表示在索引位置初始化为1
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    # 推断输出的信息，输出数据维度和类型
    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), th.float32