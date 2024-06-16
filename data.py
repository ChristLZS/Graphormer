#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:57
"""
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
import sys


# 加载 ESOL 数据集，将其划分为训练集和测试集，并返回相应的数据加载器和特征信息
def load_ESOL(args):
    # 每个样本：Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])
    # x: 节点特征矩阵，维度为[32, 9]，32个节点，每个节点9维特征
    # edge_index: 边索引，维度为[2, 68]，68条边
    # edge_attr: 边特征矩阵，维度为[68, 3]，68条边，每条边3维特征
    dataset = MoleculeNet(root="Data/MoleculeNet", name="ESOL")

    # 1128个样本用于graph-level prediction 训练：902；测试：226
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=args.seed
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=6
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=6
    )

    # Test
    # print(train_dataset[0])
    # print(dataset.num_node_features)
    # print(dataset.num_edge_features)
    # sys.exit(0)
    #
    # Data(x=[27, 9], edge_index=[2, 60], edge_attr=[60, 3], smiles='CC34CC(O)C1(F)C(CCC2=CC(=O)CCC12C)C3CCC4(O)C(=O)CO ', y=[1, 1])
    # 9
    # 3

    return (
        train_loader,
        test_loader,
        dataset.num_node_features,
        dataset.num_edge_features,
    )
