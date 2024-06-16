#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: XiaShan
@Contact: 153765931@qq.com
@Time: 2024/4/17 20:57
"""
import os, torch, random
from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim

from data import load_ESOL
from model import Graphormer
from parameter import parse_args, IOStream, table_printer


def train(args, IO, train_loader, num_node_features, num_edge_features):
    # 使用GPU or CPU
    device = torch.device(
        "cpu" if args.gpu_index < 0 else "cuda:{}".format(args.gpu_index)
    )
    if args.gpu_index < 0:
        IO.cprint("Using CPU")
    else:
        IO.cprint("Using GPU: {}".format(args.gpu_index))
        torch.cuda.manual_seed(args.seed)  # TODO 多余的？main中设置过了

    # 加载模型及参数量统计
    model = Graphormer(args, num_node_features, num_edge_features).to(device)
    IO.cprint(str(model))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    IO.cprint("Model Parameter: {}".format(total_params))

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    IO.cprint("Using AdamW")

    # 损失函数
    criterion = nn.L1Loss(reduction="sum")

    # 训练
    # trange是tqdm的一个特殊用法，用于显示循环的进度
    epochs = trange(args.epochs, leave=True, desc="Epochs")
    for epoch in epochs:
        #################
        ###   Train   ###
        #################
        model.train()  # 训练模式
        train_loss = 0.0  # 一个epoch，所有样本损失总和

        for i, data in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Train_Loader"
        ):
            data = data.to(device)  # 数据移动到GPU
            optimizer.zero_grad()  # 梯度清零

            outputs = model(data)  # 前向传播
            loss = criterion(outputs, data.y)  # 计算损失
            loss.backward()  # 反向传播

            # 剪裁可迭代参数的梯度范数，防止梯度爆炸
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新参数

            train_loss += loss.item()

        # 输出训练损失
        IO.cprint(
            "Epoch #{:03d}, Train_Loss: {:.6f}".format(
                epoch, train_loss / len(train_loader.dataset)
            )
        )

    # 保存模型
    torch.save(model, "outputs/%s/model.pth" % args.exp_name)
    IO.cprint(
        "The current best model is saved in: {}".format(
            "******** outputs/%s/model.pth *********" % args.exp_name
        )
    )


def test(args, IO, test_loader):
    """测试模型"""
    device = torch.device(
        "cpu" if args.gpu_index < 0 else "cuda:{}".format(args.gpu_index)
    )

    # 输出内容保存在之前的训练日志里
    IO.cprint("")
    IO.cprint("********** TEST START **********")
    IO.cprint("Reload Best Model")
    IO.cprint(
        "The current best model is saved in: {}".format(
            "******** outputs/%s/model.pth *********" % args.exp_name
        )
    )

    model = torch.load("outputs/%s/model.pth" % args.exp_name).to(device)
    model = model.eval()  # 创建一个新的评估模式的模型对象，不覆盖原模型

    ################
    ###   Test   ###
    ################
    test_loss = 0.0

    # 损失函数
    criterion = nn.L1Loss(reduction="sum")

    for i, data in tqdm(
            enumerate(test_loader), total=len(test_loader), desc="Test_Loader"
    ):
        data = data.to(device)  # 数据移动到GPU
        with torch.no_grad():  # 不计算梯度
            outputs = model(data)  # 前向传播
            loss = criterion(outputs, data.y)  # 计算损失

        test_loss += loss.item()

    IO.cprint("TEST :: Test_Loss: {:.6f}".format(test_loss / len(test_loader.dataset)))


def exp_init():
    """实验初始化"""
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    if not os.path.exists("outputs/" + args.exp_name):
        os.mkdir("outputs/" + args.exp_name)

    # 保存原始代码，windows下使用copy命令，且使用双引号

    # windows
    # os.system(f"copy main.py outputs\\{args.exp_name}\\main.py.backup")
    # os.system(f"copy data.py outputs\\{args.exp_name}\\data.py.backup")
    # os.system(f"copy layer.py outputs\\{args.exp_name}\\layer.py.backup")
    # os.system(f"copy model.py outputs\\{args.exp_name}\\model.py.backup")
    # os.system(f"copy parameter.py outputs\\{args.exp_name}\\parameter.py.backup")

    # linux
    os.system("cp main.py outputs" + "/" + args.exp_name + "/" + "main.py.backup")
    os.system("cp data.py outputs" + "/" + args.exp_name + "/" + "data.py.backup")
    os.system("cp layer.py outputs" + "/" + args.exp_name + "/" + "layer.py.backup")
    os.system("cp model.py outputs" + "/" + args.exp_name + "/" + "model.py.backup")
    os.system(
        "cp parameter.py outputs" + "/" + args.exp_name + "/" + "parameter.py.backup"
    )


# 相比较于原文：
# 没有构建虚拟节点，直接使用原始图结构
# 最短路径特征使用了Floyd-Warshall算法
# python main.py --exp_name=ESOL_Graphormer
if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)  # 设置Python随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    exp_init()

    IO = IOStream("outputs/" + args.exp_name + "/run.log")
    IO.cprint(str(table_printer(args)))  # 打印参数表格

    train_loader, test_loader, num_node_features, num_edge_features = load_ESOL(args)

    train(args, IO, train_loader, num_node_features, num_edge_features)
    test(args, IO, test_loader)
