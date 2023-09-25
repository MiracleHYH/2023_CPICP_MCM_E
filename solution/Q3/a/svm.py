import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv('./dataset.csv', header=None)

X_train = dataset.iloc[:100, :-1].values
y_train = dataset.iloc[:100, -1].values
X_test = dataset.iloc[100:, :-1].values
y_test = dataset.iloc[100:, -1].values

# 创建 SVM 模型
model = svm.SVC(kernel='linear')  # 选择线性 SVM，你也可以选择其他核函数

# 训练 SVM 模型
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred
np.savetxt('./answer.csv', y_pred.astype(np.int32), delimiter=',', fmt='%d')
# # 计算准确度
# accuracy = accuracy_score(y_test, y_pred)
# print("准确度:", accuracy)
#
# # 打印分类报告
# print(classification_report(y_test, y_pred))
#
# # 打印混淆矩阵
# conf_matrix = confusion_matrix(y_test, y_pred)
# print("混淆矩阵:\n", conf_matrix)
