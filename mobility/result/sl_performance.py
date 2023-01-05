import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({'font.size': 25})
plt.rc('axes', labelsize=30)
# plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()

#delay
# train_data = pd.read_csv("./last/delay_flow_amount.csv", names=["amount", "delay", "protocol"])
train_data = pd.read_csv("result(all).csv", names=["path", "error", 'theta'])

ax = sns.barplot(x="path", y="error", data=train_data, palette="plasma")
for p in ax.patches :
 print(p.get_height())
print("+++")
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

ax.yaxis.set_major_locator(MultipleLocator(0.5))
# # ax.yaxis.set_major_formatter('{x:.001f}') ## 메인 눈금이 표시될 형식
ax.yaxis.set_minor_locator(MultipleLocator(0.25))
ax.tick_params(axis='x',which='major',width=3, size=6, direction='out')
ax.tick_params(axis='y',which='major',width=3, size=6, direction='in')
# ax.tick_params(axis='x',which='minor',width=2, size=6, direction='in')
ax.tick_params(axis='y',which='minor',width=2, size=6, direction='in')
for i in ax.get_xticklabels() : i.set_fontweight("bold")
for i in ax.get_yticklabels() : i.set_fontweight("bold")
# ax.legend(leg_handles, [ "\u03C9=2, \u03C9'=4",
#                          "\u03C9=1, \u03C9'=4",
#                          "\u03C9=1, \u03C9'=3"], title='')

# plt.ylim(0.5, 1.9)

ax.grid(axis='y')
ax.grid(axis='y', which='minor')
ax.set_axisbelow(True)
plt.xlabel("Length of prediction", labelpad=10,fontweight='bold')
plt.ylabel("Distance error (cell)", labelpad=10,fontweight='bold')

plt.savefig('trajectory_prediction_error.pdf', bbox_inches='tight')
# plt.show()
