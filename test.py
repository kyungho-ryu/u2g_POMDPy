import numpy as np
import torch
import matplotlib.pyplot as plt
def test(x) :
    # return (2/(1+np.exp(-2*x))) -1
    # return np.exp(3*x)
    return (24 * x + 24)/2

x = np.arange(-1,1.1,0.1)
print(x)
# print(x)
# y = torch.tanh(torch.from_numpy(x).float())
# print(y)
# print(24**y))
plt.plot(x, test(x))
plt.show()