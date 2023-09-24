import pandas as pd
import os
from conf import dataPath
from datetime import datetime


def get_time(time):
    return datetime.strptime(time, '%Y/%m/%d %H:%M')


table1 = pd.read_csv(os.path.join(dataPath, 'table1.csv'))
table2 = pd.read_csv(os.path.join(dataPath, 'table2.csv'))
table_time = pd.read_csv(os.path.join(dataPath, 'serial_vs_time.csv'))
columns = [table2.columns.get_loc('首次检查流水号')]
for follow_time in range(1, 8):
    columns.append(table2.columns.get_loc(f'随访{follow_time}流水号'))

data_extracted = []
mixed_data = []
for i in range(160):
    row = [table2.iloc[i, 0]]
    diagnose_duration = table1.iloc[i, 14]
    time_start = get_time(table_time.iloc[i, 2])
    ed_start = table2.iloc[i, columns[0] + 12]
    tag_i = 0
    for j in range(8):
        idx_time = 2 + j * 2
        idx_v = columns[j] + 12
        if pd.isnull(table2.iloc[i, idx_v]):
            row.append(None)
            continue
        dt = get_time(table_time.iloc[i, idx_time]) - time_start
        dt_days = ((dt.total_seconds() / 3600) + diagnose_duration) / 24
        dv_absolute = table2.iloc[i, idx_v] - ed_start
        dv_rate = dv_absolute / ed_start
        row.append((dt_days, ed_start / 1000, dv_absolute / 1000, dv_rate * 100))
        mixed_data.append([diagnose_duration, dt.total_seconds() / 3600 + diagnose_duration, ed_start, dv_absolute])
    data_extracted.append(row)
pd.DataFrame(data_extracted, columns=['id', 'first'] + [f'follow_{i}' for i in range(1, 8)]).to_csv(
    './table2_extracted.csv', index=False)
pd.DataFrame(mixed_data, columns=['diagnose_duration', 'dt', 'v_start', 'v_t']).to_csv('./mixed_data.csv', index=False)
