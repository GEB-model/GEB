import matplotlib.pyplot as plt
import pandas as pd


NAME = "Wadenepally"
LOCATION = (80.0731, 16.7942)


df = pd.read_csv('calibration/observations.csv', parse_dates=['Dates']).rename({'Current Year (Flow in cumecs)': 'flow'}, axis=1)
n = len(df)
df = df[df['flow'] != '-']
n_not_dropped = len(df)
print(f"Dropped {n-n_not_dropped} out of {n} rows because data is not available.")
df['flow'] = df['flow'].astype(float)

plt.plot(df['Dates'], df['flow'])
plt.show()