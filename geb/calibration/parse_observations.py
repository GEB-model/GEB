import matplotlib.pyplot as plt
import pandas as pd
import os


CALIBRATION_FOLDER = os.path.join('DataDrive', 'GEB', 'calibration')
NAME = "Wadenepally"
LOCATION = (80.0731, 16.7942)


df = pd.read_csv(
    os.path.join(CALIBRATION_FOLDER, 'observations', f'{NAME}.csv'), parse_dates=['Dates']
).rename({'Current Year (Flow in cumecs)': 'flow'}, axis=1)

n = len(df)
df = df[df['flow'] != '-']
n_not_dropped = len(df)
print(f"Dropped {n-n_not_dropped} out of {n} rows because data is not available.")
df['flow'] = df['flow'].astype(float)

df = df[['Dates', 'flow']]
df.to_csv(os.path.join(CALIBRATION_FOLDER, 'observations.csv'), index=False)

plt.plot(df['Dates'], df['flow'])
plt.show()