import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import exp


def f_gauss(x, params):
    [a, b, c] = params
    y = []
    for _x in x:
        y.append(a * exp(-((_x - b) / c) ** 2))
    return y


def main():
    df = pd.read_csv('./tagged_points.csv')
    params_list = [
        [1638, 1546, 881.6],
        [30.49, 251.4, 445.3],
        [42.41, 377.8, 430.9],
        [40.31, 374.6, 503.2],
        [38.33, 445.3, 400.8]
    ]
    color_map = plt.cm.get_cmap('viridis', 5)
    colors = [color_map(i / (5 - 1)) for i in range(5)]
    for _, row in df.iterrows():
        [__, dt, v_ed, person_class] = row.values
        dt = float(dt)
        v_ed = float(v_ed)
        person_class = int(person_class)
        c = params_list[person_class]
        plt.scatter(dt, v_ed, c=colors[person_class])
    for param in params_list:
        x = np.linspace(0, 200, 1000)
        plt.plot(x, f_gauss(x, param), c=colors[params_list.index(param)],
                 label=f'person_class={params_list.index(param)}')
    plt.xlabel('dt')
    plt.ylabel('v_ed')
    plt.legend()
    plt.savefig('plot.png')
    plt.savefig('plot.pdf')
    plt.show()


if __name__ == '__main__':
    main()
