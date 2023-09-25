import pandas as pd
import os
from conf import dataPath
import numpy as np

table1 = pd.read_csv(os.path.join(dataPath, 'table1.csv'))
table2 = pd.read_csv(os.path.join(dataPath, 'table2.csv'))
table3_ed = pd.read_csv(os.path.join(dataPath, 'table3_ed.csv'))
table3_hm = pd.read_csv(os.path.join(dataPath, 'table3_hemo.csv'))

self_info = table1.iloc[:, 4:22].drop(columns=['血压'])
pressure = table1['血压'].str.split('/', expand=True)
mRS = table1['90天mRS'].fillna(-1).astype(np.int32)
self_info['性别'] = self_info['性别'].map({'男': 1, '女': 0})

vp_info = table2.iloc[:, 2:23]

sd_info = []
for i in range(160):
    serial_number = table1.iloc[i, 3]
    sd_info.append(
        np.concatenate((
            table3_ed[table3_ed['流水号'] == serial_number].iloc[0, 2:32].values,
            table3_hm[table3_hm['流水号'] == serial_number].iloc[0, 2:32].values
        ))
    )
x = pd.DataFrame(
    np.concatenate((
        self_info.values,
        pressure.values,
        vp_info.values,
        np.array(sd_info)
    ), axis=1).astype(np.float32)
)
x = x.fillna(0)
x = (x - x.min()) / (x.max() - x.min())
dataset = pd.DataFrame(np.concatenate((x.values, mRS.values.reshape(-1, 1)), axis=1))
dataset.to_csv('./dataset.csv', index=False, header=False)
