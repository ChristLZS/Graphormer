#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:41
"""
from typing import Union, Tuple, Dict, List

import torch
import networkx as nx
import sys
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils.convert import to_networkx

from layer import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding


# 遍历图,查询从 source 节点到所有节点的最短路径
def floyd_warshall_source_to_all(G, source, cutoff=None):
    "Floyd-Warshall算法查询最短路径(BFS遍历图)"
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if cutoff is not None and cutoff <= level:
            break

    # Test
    # print(node_paths)
    # print(edge_paths)
    # sys.exit(0)
    # 0: [0], 1: [0, 1], 10: [0, 10], 2: [0, 1, 2], 9: [0, 10, 9], 11: [0, 10, 11], 3: [0, 1, 2, 3], 8: [0, 10, 9, 8], 6: [0, 10, 11, 6], 4: [0, 1, 2, 3, 4], 7: [0, 10, 9, 8, 7], 5: [0, 10, 11, 6, 5]}
    # {0: [], 1: [0], 10: [1], 2: [0, 3], 9: [1, 23], 11: [1, 24], 3: [0, 3, 5], 8: [1, 23, 20], 6: [1, 24, 26], 4: [0, 3, 5, 8], 7: [1, 23, 20, 18], 5: [1, 24, 26, 13]}

    return node_paths, edge_paths


# 遍历图,查询所有节点对之间的最短路径
def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
    node_paths = {n: paths[n][0] for n in paths}
    edge_paths = {n: paths[n][1] for n in paths}
    return node_paths, edge_paths


def shortest_path_distance(
    data: Data,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths


# 批量获取最短路径数据
def batched_shortest_path_distance(
    data,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    graphs = [to_networkx(sub_data) for sub_data in data.to_data_list()]
    relabeled_graphs = []  # 重新标记节点的图
    shift = 0  # 节点偏移量
    for i in range(len(graphs)):
        num_nodes = graphs[i].number_of_nodes()
        relabeled_graphs.append(
            nx.relabel_nodes(graphs[i], {i: i + shift for i in range(num_nodes)})
        )
        shift += num_nodes

    paths = [all_pairs_shortest_path(G) for G in relabeled_graphs]
    node_paths = {}
    edge_paths = {}

    for path in paths:
        for k, v in path[0].items():
            node_paths[k] = v
        for k, v in path[1].items():
            edge_paths[k] = v

    return node_paths, edge_paths


# Graphormer模型,继承自nn.Module,需要实现 __init__ 和 forward 方法
# num_node_features: 节点特征的维度
# num_edge_features: 边特征的维度
# 以上两个参数根据数据集的特征维度来确定
class Graphormer(nn.Module):
    def __init__(self, args, num_node_features, num_edge_features):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()  # 调用父类的构造函数

        # 初始化参数
        self.num_layers = args.num_layers
        self.input_node_dim = num_node_features
        self.node_dim = args.node_dim
        self.input_edge_dim = num_edge_features
        self.edge_dim = args.edge_dim
        self.output_dim = args.output_dim
        self.num_heads = args.num_heads
        self.max_in_degree = args.max_in_degree
        self.max_out_degree = args.max_out_degree
        self.max_path_distance = args.max_path_distance

        # 创建节点特征的输入线性层和边特征的输入线性层
        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        # 创建中心性编码和空间编码
        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim,
        )
        self.spatial_encoding = SpatialEncoding(
            max_path_distance=self.max_path_distance,
        )

        # 创建Graphormer注意力层
        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    num_heads=self.num_heads,
                    max_path_distance=self.max_path_distance,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 初始化节点输出线性层
        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    # 前向传播,data是一个Data对象,包含了图的信息
    # 返回值是一个torch.Tensor,表示节点的输出特征
    def forward(self, data: Union[Data]) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()  # 边索引
        edge_attr = data.edge_attr.float()  # 边特征

        if type(data) == Data:  # 如果data是单个图
            ptr = None
            # 最短路径特征
            node_paths, edge_paths = shortest_path_distance(data)
        else:  # 如果data是一个batch of graphs
            ptr = data.ptr
            # 最短路径特征
            node_paths, edge_paths = batched_shortest_path_distance(data)

        # 输入特征线性变换
        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        # 中心性编码和空间编码
        x = self.centrality_encoding(x, edge_index)
        b = self.spatial_encoding(x, node_paths)

        # Graphormer层,多层堆叠,每一层的输入是上一层的输出
        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        # 输出特征线性变换
        x = self.node_out_lin(x)

        # 全局平均池化,因为encoder的输出是节点的特征,需要将其池化为图的特征
        x = global_mean_pool(x, data.batch)

        return x
