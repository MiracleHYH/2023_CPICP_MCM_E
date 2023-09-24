import matplotlib.pyplot as plt
import numpy as np

# 示例数据，包含[value, tag]的列表
data = [[1.2, 0],
        [2.0, 1],
        [2.5, 1],
        [3.0, 2],
        [3.2, 2],
        [3.5, 2],
        [4.0, 3],
        [4.2, 3],
        [4.5, 4]]

# 将数据按照tag值分组
grouped_data = {}
for value, tag in data:
    if tag not in grouped_data:
        grouped_data[tag] = []
    grouped_data[tag].append(value)

# 提取不同tag值和对应的数据
tags = list(grouped_data.keys())
values = list(grouped_data.values())

# 绘制箱线图
plt.boxplot(values, labels=tags)

# 添加标题和标签
plt.title('Box Plot by Tag')
plt.xlabel('Tag')
plt.ylabel('Value')

# 显示图形
plt.show()
