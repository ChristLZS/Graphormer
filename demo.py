import torch
import torch.nn as nn


class ExampleModel(nn.Module):
    def __init__(self, max_in_degree, max_out_degree, node_dim):
        super(ExampleModel, self).__init__()
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x, in_degree, out_degree):
        t = self.z_in[in_degree]
        x += self.z_in[in_degree] + self.z_out[out_degree]
        return x


# 假设 max_in_degree = 5, max_out_degree = 5, node_dim = 10
model = ExampleModel(max_in_degree=5, max_out_degree=5, node_dim=10)

# 假设输入特征 x, 入度索引 in_degree 和出度索引 out_degree
x = torch.zeros((1, 10))  # 形状为 (1, 10) 的零张量
in_degree = 2  # 假设入度为 2
out_degree = 3  # 假设出度为 3

# 前向传播
output = model(x, in_degree, out_degree)
print(output)
