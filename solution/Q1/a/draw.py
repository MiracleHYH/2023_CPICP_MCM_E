import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import numpy as np


def linear_model(params, x):
    m, b = params
    return m * x + b


# 定义 bisquare 权重函数
def bisquare_weight(r):
    c = 4.685  # Huber M-estimator consistency constant
    if abs(r) <= c:
        return (1 - (r / c) ** 2) ** 2
    else:
        return 0


# 定义损失函数，使用 bisquare 权重
def loss_function(params, x, y):
    residuals = y - linear_model(params, x)
    weights = np.array([bisquare_weight(r) for r in residuals])
    return weights * residuals


# 初始参数猜测
initial_guess = (1.0, 1.0)


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
            # plt.plot(x, y, c='b')
    X = np.array(X)
    Y = np.array(Y)
    # params, covariances = curve_fit(linear_model, X, Y)
    # m, b = params
    result = least_squares(loss_function, initial_guess, args=(X, Y))

    # 获取拟合的参数值
    m, b = result.x
    print("斜率 m:", m)
    print("截距 b:", b)
    #
    # # 绘制拟合曲线和原始数据
    plt.scatter(X, Y, label='Data', color='blue')
    plt.plot(X, m * X + b, label='Fitted Line', color='red')
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
    t_rate = (1.33 - b) / m
    answers = []
    for i in range(100):
        if tag.iloc[i, 0] == 0:
            answers.append(0)
            continue
        (_, v_start, _, _) = ast.literal_eval(df.iloc[i, 1])
        t_v = (6 / v_start + 1 - b) / m
        t_answer = t_rate if t_rate < t_v else t_v
        answers.append(t_answer if t_answer < 2.0 else 0)
    pd.DataFrame(answers, columns=['predict_t']).to_csv('./answer.csv', index=False)


if __name__ == '__main__':
    main()
