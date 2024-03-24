import random

import torch
import mujoco
import numpy as np
from numpy.random import random
from scipy.optimize import minimize

# import plt
import matplotlib.pyplot as plt
from scipy import optimize

# from scipy.optimize import optimize

# from sklearn.utils import optimize

# from scipy import optimize

# This is an implemtation of a simple question in ppt //todo page

# generate b and truePos
b = np.mat([[1, -2, -4, 5, -7, 8], [3, -2, 3, 6, 5.5, 10]])
truePos = np.transpose(np.mat([4, 0]))

plt.plot(b[0, :], b[1, :], 'ro')
plt.plot(truePos[0], truePos[1], 'bo')
plt.show()
# print("pipupipu chapachapachapa")

def cost233(theta, b, y):
    '''

    :param theta:
    :param b:
    :param y:
    :return:
    '''
    j = 0
    # m = np.size(b, 1)
    m = b.shape[1]
    for i in range(m):
        # axis = 0 -> column vector;
        # axis = 1 -> row vector
        j += np.square(y[i] - np.linalg.norm(theta - b[:, i], axis=0))
    print("j: ", j)
    return j[0]

def cost(theta, b, y):
    m = b.shape[1]
    res = 0
    for i in range(m):
        res += np.square(y[i] - np.linalg.norm(theta - b[:, i]))
    print("res: ", res)
    return res

def cost(theta, b, y):
    m = b.shape[1] # Get the number of columns
    res = np.sum(np.square(y - np.linalg.norm(theta[:, None] - b, axis=0)))
    print("res: ", res)
    return res



# generate mesuarment data
m = np.size(b, 1)

# generate noise
noise = 0.1 * random(m)
y = np.zeros(m)

for i in range(m):
    y[i] = np.linalg.norm(truePos - b[:, i], axis=0) + noise[i]

# init theta
# init theta
theta0 = np.array([0, 0])

# least squares to get the result
thetahat_LS = np.linalg.lstsq(b.T, y, rcond=None)[0]

print("Estimated position (Least Squares): ", thetahat_LS)

# optimize the result
res = minimize(cost, thetahat_LS, args=(b, y), method='BFGS')

# use theta_LS to calculate the result

# plot the result
plt.plot(b[0, :], b[1, :], 'ro')
plt.plot(truePos[0], truePos[1], 'bo')
# plt.plot(thetahat_LS[0], thetahat_LS[1], 'go')
plt.title('Estimated position')
plt.plot(res.x[0], res.x[1], 'yo')
plt.show()
