import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


def linear_model(x, m, b):
    return m * x + b


def main():
    df = pd.read_csv('./table2_extracted.csv')
    tag = pd.read_csv('./tag.csv')
    # fig, axes = plt.subplots(4, 8, figsize=(8, 10))
    # nr = 0
    # nc = 0
    # count = 0
    X = []
    Y = []
    for i in range(100):
        if tag.iloc[i, 0] == 1:
            # ax = axes[nr][nc]
            # nc += 1
            # if nc >= 8:
            #     nr += 1
            #     nc = 0
            # count += 1
            # ax.set_title(f'No.{i + 1}')
            x = []
            y = []
            for j in range(1, 9):
                if not pd.isnull(df.iloc[i, j]):
                    (dt, v_start, dv_absolute, dv_rate) = ast.literal_eval(df.iloc[i, j])
                    if dt > 2.0:
                        break
                    X.append(dt)
                    Y.append((v_start + dv_absolute) / v_start)
                    x.append(dt)
                    y.append((v_start + dv_absolute) / v_start)
                    # if dt < 2.0:
                    #     # if v_absolute > 6 or v_rate > 33:
                    #     # plt.axhline(y=6, color='red', linestyle='--', label='y=6')
                    #     plt.scatter(dt, v_rate, c='b')
                    #     # plt.axhline(y=30, color='red', linestyle='--', label='y=30')
                    #     # ax.scatter(dt, v_start, v_absolute, c='b')
                    #     # break
            plt.plot(x, y, c='b')
    X = np.array(X)
    Y = np.array(Y)
    params, covariances = curve_fit(linear_model, X, Y)
    m, b = params
    print("斜率 m:", m)
    print("截距 b:", b)

    # 绘制拟合曲线和原始数据
    # plt.scatter(X, Y, label='Data', color='blue')
    plt.plot(X, linear_model(X, m, b), label='Fitted Line', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()
    # ax.set_xlabel('dt')
    # ax.set_ylabel('v_start')
    # ax.set_zlabel('v_absolute')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()
