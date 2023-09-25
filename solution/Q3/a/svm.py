import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

dataset = pd.read_csv('./dataset.csv', header=None)

X_train = dataset.iloc[:80, :-1].values
y_train = dataset.iloc[:80, -1].values
X_test = dataset.iloc[80:100, :-1].values
y_test = dataset.iloc[80:100, -1].values


mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# model = svm.SVC(kernel='linear')
#
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# # np.savetxt('./answer.csv', y_pred.astype(np.int32), delimiter=',', fmt='%d')
#

accuracy = accuracy_score(y_test, y_pred)
print("准确度:", accuracy)

print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", conf_matrix)
