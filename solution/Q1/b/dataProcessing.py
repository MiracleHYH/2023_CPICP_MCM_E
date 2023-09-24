import os

import numpy as np
import pandas as pd
from conf import dataPath

# E-W 4-22
table1 = pd.read_csv(os.path.join(dataPath, 'table1.csv'))
# C-X 2-23
table2 = pd.read_csv(os.path.join(dataPath, 'table2.csv'))
# C-AG 2-32
table3 = pd.read_csv(os.path.join(dataPath, 'table3_hemo.csv'))
# tag
tag = pd.read_csv('../a/tag.csv')

dataset = []
for i in range(160):
    [pressure_high, pressure_low] = table1.iloc[i, 15].split('/')
    table1_info = np.concatenate(
        (table1.iloc[i, 4:14].values, np.array([pressure_high, pressure_low]), table1.iloc[i, 16:22].values))

    table2_info = table2.iloc[i, 2:23].values

    serial_number = table1.iloc[i, 3]
    table3_info = table3[table3['流水号'] == serial_number].iloc[0, 2:32].values

    row = np.concatenate((table1_info, table2_info, table3_info, tag.iloc[i]))
    dataset.append(row)
df = pd.DataFrame(dataset)
df[1] = df[1].map({'男': 1, '女': 0})
df = df.astype(np.float32)
df_normalized = (df - df.min()) / (df.max() - df.min())
df_normalized.to_csv('./dataset.csv', index=False, header=False)
