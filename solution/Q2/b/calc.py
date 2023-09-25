from math import exp
import pandas as pd
import numpy as np
import ast


def f_gauss(x, params):
    [a, b, c] = params
    return a * exp(-((x - b) / c) ** 2)


def residual(x, y, f, params):
    return abs(y - f(x, params))


def main():
    params_list = [
        [1638, 1546, 881.6],
        [30.49, 251.4, 445.3],
        [42.41, 377.8, 430.9],
        [40.31, 374.6, 503.2],
        [38.33, 445.3, 400.8]
    ]
    df = pd.read_csv('./tagged_points.csv')
    answer = []
    for i in range(100):
        answer.append(0)
    for _, row in df.iterrows():
        [person_id, dt, v_ed, person_class] = row.values
        person_id = int(person_id)
        dt = float(dt)
        v_ed = float(v_ed)
        person_class = int(person_class)
        answer[person_id] += residual(dt, v_ed, f_gauss, params_list[person_class])
    pd.DataFrame(answer).to_csv('./residual.csv', index=False, header=False)


if __name__ == '__main__':
    main()
