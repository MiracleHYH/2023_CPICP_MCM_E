import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from MLP import MLP

# 读取数据
df = pd.read_csv('./mixed_data.csv')

df_48 = df[df['dt'] < 48]

X_train = df_48[['diagnose_duration', 'dt', 'v_start']].values
y_train = df_48[['v_t']].values.reshape(-1, 1)

# 转换数据为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)


# 定义模型参数
input_size = 3
hidden_size = 64
output_size = 1

# 创建模型实例
model = MLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练模型
epochs = 100000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train)

    # 计算损失
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# 模型训练完成后，你可以使用它进行预测
# 假设 X_test 是测试数据，你可以将其转换为PyTorch张量，然后使用模型进行预测
# X_test = torch.FloatTensor(X_test)
# predictions = model(X_test)
