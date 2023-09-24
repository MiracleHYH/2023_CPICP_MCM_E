import pandas as pd
import os
from conf import dataPath
from datetime import datetime


def get_time(time):
    return datetime.strptime(time, '%Y/%m/%d %H:%M')


person_class = pd.read_csv('./class.csv', header=None)
table1 = pd.read_csv(os.path.join(dataPath, 'table1.csv'))
table2 = pd.read_csv(os.path.join(dataPath, 'table2.csv'))
table_time = pd.read_csv(os.path.join(dataPath, 'serial_vs_time.csv'))
columns = [table2.columns.get_loc('首次检查流水号')]
for follow_time in range(1, 8):
    columns.append(table2.columns.get_loc(f'随访{follow_time}流水号'))

tagged_data = []
for i in range(100):
    row = [table2.iloc[i, 0]]
    diagnose_duration = table1.iloc[i, 14]
    time_start = get_time(table_time.iloc[i, 2])
    hm_start = table2.iloc[i, columns[0] + 12]
    for j in range(8):
        idx_time = 2 + j * 2
        idx_v = columns[j] + 12
        if not pd.isnull(table2.iloc[i, idx_v]):
            dt = get_time(table_time.iloc[i, idx_time]) - time_start
            dt_days = ((dt.total_seconds() / 3600) + diagnose_duration) / 24
            tagged_data.append([dt_days, table2.iloc[i, idx_v] / 1000, person_class.iloc[i, 0]])
pd.DataFrame(tagged_data, columns=['dt', 'v_ed', 'class']).to_csv('./tagged_points.csv', index=False)
