import torch
import torch.nn as nn


# 创建神经网络模型
class BinaryClassificationModel(nn.Module):
    def __init__(self):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(69, 8)  # 输入特征数为69，隐藏层神经元数为64
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)  # 输出层有一个神经元，用于二分类

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
