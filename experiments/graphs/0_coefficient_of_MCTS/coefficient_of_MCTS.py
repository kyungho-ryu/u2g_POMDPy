import numpy as np
from past.utils import old_div
import matplotlib.pyplot as plt

ucb_coefficient = 1
N = 10

x = np.linspace(1, 10, 10)
y = ucb_coefficient * np.sqrt(old_div(np.log(N + 1), x))

plt.plot(x, y)
plt.show()
