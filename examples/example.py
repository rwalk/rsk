import os
import sys
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import matrix

# need to load parent module to run this script
sys.path.append("..")
from rsk.rsk import RSK

def example():

    # randomly chosen translation matrix
    Z = matrix([[1.2,.3]])

    # a slow moving transition matrix
    F = matrix([[0.99, 0.01], [0.01,0.99]])
    a0 = np.matrix([1,0]).transpose()

    #
    # generate data according to the alpha process
    #

    # true alpha/means
    true_alpha = np.zeros((10,2,1))
    true_alpha[0] = a0
    for i in range(1,10):
        true_alpha[i] = F.dot(true_alpha[i-1])
    true_means = [Z.dot(a)[0].A1[0] for a in true_alpha]

    # observed mu
    Q = 0.01*sp.eye(2)
    alpha = np.zeros((10,2,1))
    alpha[0] = a0
    for i in range(1,10):
        alpha[i] = F.dot(alpha[i-1]) + np.random.multivariate_normal([0,0], Q).reshape((-1,1))

    mu = [Z.dot(a)[0].A1[0] for a in alpha]

    # generated data
    y = np.random.multivariate_normal(mu, 0.05*sp.eye(10), size=(1,200)).transpose()
    raw_means = y.mean(axis=1)

    # fit mu with RSK
    rsk = RSK(F,Z)
    fitted_means = rsk.fit(y, sp.eye(1), a0, Q, Q)
    smoothed_fitted_means = rsk.fit(y, sp.eye(1), a0, Q, Q, smooth=True)

    #
    # plot figure
    #
    fig = plt.figure()
    t=[i+1 for i in range(10)]
    plt.scatter(np.repeat(sp.matrix(t), 200, axis=1).A1, y.reshape((-1,)), linewidths=0.01, s=12, c="r", label="data")
    plt.scatter(t, true_means, linewidths=0.25, s=72, c="yellow",label="true mean")
    plt.scatter(t, fitted_means, linewidths=0.25, s=72, c="blue", label="rsk mean")
    #plt.scatter(t, smoothed_fitted_means, linewidths=0.25, s=64, c="green", label="rsk smoothed mean")
    plt.scatter(t, raw_means, linewidths=0.25, s=72, c="gray", label="raw mean")
    plt.legend()
    plt.title("Repeated Surveys Kalman Filter Demo")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()

if __name__ == "__main__":
    example()