# import numpy as np
#
# mu = 0
# logstd = 0.1
# for i in range(10) :
#     print(np.random.normal())
# # for i in range(100) :
# #     a = mu + np.exp(logstd) * np.random.normal()
# #     print(a)

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns
# mu = 0
# variance = 1
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))
# plt.show()

# value = np.random.normal(loc=0,scale=1,size=1000)
# plt.boxplot(x)

x = np.linspace(-1, 1, 100)
y = []
for i in range(len(x)) :
    y.append(np.exp(x[i]))
plt.plot(x, y)
plt.show()
# print(value)
# sns.distplot(value)