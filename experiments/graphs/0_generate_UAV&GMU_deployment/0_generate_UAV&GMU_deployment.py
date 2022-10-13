import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

file = str(pathlib.Path().resolve()) + '/csvFile/GMU_deployment.csv'
gmu_deployment = pd.read_csv(file)
df = gmu_deployment.pivot('y', 'x', 'count')

print(df)
h = 226
s = 0.70
v = 1

colors = [
    mcl.hsv_to_rgb((h / 360, 0, v)),
    mcl.hsv_to_rgb((h / 360, 0.25, v)),
    mcl.hsv_to_rgb((h / 360, 0.5, v)),
    mcl.hsv_to_rgb((h / 360, 0.75, v)),
    mcl.hsv_to_rgb((h / 360, 1, v))
]
cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

sns.heatmap(df, cmap=cmap)

file = str(pathlib.Path().resolve()) + '/csvFile/UAV_deployment.csv'
df = pd.read_csv(file)
plt.scatter(df["x"], df["y"], c='red', marker='x', s=20*2)

plt.gca().invert_yaxis()
file = str(pathlib.Path().resolve()) + '/imgs/deployment(T=2074).pdf'
plt.savefig(file)
plt.show()
