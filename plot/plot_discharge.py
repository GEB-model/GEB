from hydro_stats import get_discharge
import matplotlib.pyplot as plt

dates, values = get_discharge('spinup')

plt.plot(dates, values)
plt.show()