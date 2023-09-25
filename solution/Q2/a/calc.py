from math import exp
import pandas as pd
import numpy as np
import ast


def f_gauss(x):
    a1 = 30.62
    b1 = 328.1
    c1 = 493.5
    return a1 * exp(-((x - b1) / c1) ** 2)


def f_poly(x):
    p = [
        8.311e-28,
        -1.618e-23,
        1.288e-19,
        -5.424e-16,
        1.309e-12,
        -1.858e-09,
        1.573e-06,
        -0.0008095,
        0.2101,
        18.14
    ]
    y = 0
    for i in range(10):
        y += p[i] * x ** (9 - i)
    return y


def residual(x, y, f):
    return abs(y - f(x))


def main():
    df = pd.read_csv('./table2_extracted.csv')
    answer = []
    for index, row in df.iterrows():
        res_gauss = 0
        res_poly = 0
        for i in range(1, 9):
            value = row.iloc[i]
            if np.str_(value) == 'nan':
                continue
            (dt, v0, dv, _) = ast.literal_eval(value)
            res_gauss += residual(dt, v0 + dv, f_gauss)
            res_poly += residual(dt, v0 + dv, f_poly)
        answer.append([res_gauss, res_poly])
    pd.DataFrame(answer).to_csv('./residual.csv', index=False, header=False)


if __name__ == '__main__':
    main()
