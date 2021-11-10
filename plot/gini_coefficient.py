import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# ensure your arr is sorted from lowest to highest values first!
arr = np.array([1000, 1,4,6,9,100])

class Gini:
    def __init__(self, arr):
        self.arr = np.sort(arr)

    def __repr__(self):
        count = self.arr.size
        coefficient = 2 / count
        indexes = np.arange(1, count + 1)
        weighted_sum = (indexes * self.arr).sum()
        total = self.arr.sum()
        constant = (count + 1) / count
        return str(coefficient * weighted_sum / total - constant)

    def lorenz(self):
        # this divides the prefix sum by the total sum
        # this ensures all the values are between 0 and 1.0
        scaled_prefix_sum = self.arr.cumsum() / self.arr.sum()
        # this prepends the 0 value (because 0% of all people have 0% of all wealth)
        return np.insert(scaled_prefix_sum, 0, 0)

g = Gini(arr)
lorenz_curve = g.lorenz()

sns.set_style("white")

fig, ax = plt.subplots(1, 1)
plt.subplots_adjust(left=0.05, right=0.88, top=0.93)
# we need the X values to be between 0.0 to 1.0
ax.plot(np.linspace(0.0, 100.0, lorenz_curve.size), lorenz_curve * 100, label="Observed")
# plot the straight line perfect equality curve
ax.plot([0,100], [0,100], label="Line of equality")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_ylabel("Cumulative share of access to water")
ax.set_xlabel("Cumulative share of farmers from lowest to highest access to water")
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("Gini coefficient")
ax.legend(frameon=False)

plt.show()