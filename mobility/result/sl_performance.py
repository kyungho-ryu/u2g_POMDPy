import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle
plt.rcParams.update({'font.size': 25})
plt.rc('axes', labelsize=28)
plt.figure(figsize=(10, 8))
# 화면 스타일 설정하기
sns.set_style("whitegrid")
#delay
# train_data = pd.read_csv("./last/delay_flow_amount.csv", names=["amount", "delay", "protocol"])
train_data = pd.read_csv("result(H=100).csv", names=["path", "error", 'theta'])

ax = sns.barplot(x="path", y="error", data=train_data, palette="Blues_d")
hatches = cycle(['','//',"\\\\"])
print(train_data.path.unique())
num_locations = len(train_data.path.unique())
print(num_locations)
for i, patch in enumerate(ax.patches):
    # Boxes from left to right
    if i % num_locations == 0:
        hatch = next(hatches)
    patch.set_hatch(hatch)

leg_handles = ax.get_legend_handles_labels()[0]
# ax.legend(leg_handles, [ "\u03C9=2, \u03C9'=4",
#                          "\u03C9=1, \u03C9'=4",
#                          "\u03C9=1, \u03C9'=3"], title='')

plt.xlabel("Length of prediction")
plt.ylabel("Distance error (cell)")
plt.savefig('mobility_prediction_error(H=100).pdf')
plt.show()