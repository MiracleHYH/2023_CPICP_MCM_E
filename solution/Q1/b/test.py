import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from Model import BinaryClassificationModel

data = pd.read_csv('./dataset.csv', header=None)

X = data.iloc[:, :-1].values  # 取特征
Y = data.iloc[:, -1].values  # 取标签

X_train = torch.tensor(X[:100, :], dtype=torch.float32)
Y_train = torch.tensor(Y[:100], dtype=torch.float32)

X_test = torch.tensor(X[100:, :], dtype=torch.float32)
Y_test = torch.tensor(Y[100:], dtype=torch.float32)
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


model = BinaryClassificationModel()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二进制交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train)

    # 计算损失
    loss = criterion(outputs, Y_train.view(-1, 1))  # 将标签Y的形状调整为与模型输出一致

    # 反向传播和参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# 预测概率
with torch.no_grad():
    probabilities = model(X_test)

error = 0
total = 60
k = 0.95
for i in range(60):
    result = 0 if probabilities[i][0].item() < k else 1
    if int(Y_test[i].item()) == 1:
        print('%.2f' % probabilities[i][0].item())
    if result != int(Y_test[i].item()):
        error += 1
print(f'Accuracy: {(1 - error / total) * 100}%')
