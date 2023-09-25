import os

import numpy as np
import pandas as pd
from conf import dataPath

table1 = pd.read_csv(os.path.join(dataPath, 'table1.csv'))
table1_info = table1.iloc[:, 4:22].drop(columns=['血压'])
pressure = table1['血压'].str.split('/', expand=True)
table1_info['血压_高'] = pressure[0].astype(np.int32)
table1_info['血压_低'] = pressure[1].astype(np.int32)
table1_info['性别'] = table1_info['性别'].map({'男': 1, '女': 0})
table1_info_norm = (table1_info - table1_info.min()) / (table1_info.max() - table1_info.min())

table2 = pd.read_csv(os.path.join(dataPath, 'table2.csv'))
table2_info = []
table2_min = []
table2_max = []
for i in range(8):
    table2_info.append(table2.iloc[:, 2 + i * 23: 2 + i * 23 + 11].values.astype(np.float32))
    print(table2_info[i])
    table2_min_i = table2_info[i].min()
    table2_max_i = table2_info[i].max()
    print(table2_min_i, table2_max_i)

table3_ed = pd.read_csv(os.path.join(dataPath, 'table3_ed.csv'))
table3_hm = pd.read_csv(os.path.join(dataPath, 'table3_hemo.csv'))
