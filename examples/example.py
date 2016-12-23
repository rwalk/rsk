import os
import sys
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import matrix
from scipy.stats import gaussian_kde

# need to load parent module to run this script
sys.path.append("..")
from rsk.rsk import RSK
from rsk.panel import PanelSeries

def jitter(arr):
    err = .002*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * err

def trial():
    # randomly chosen translation matrix
    Z = matrix([[1.2,.3]])

    # a slow moving transition matrix
    F = matrix([[0.99, 0.01], [0.01,0.99]])
    a0 = np.matrix([1,0]).transpose()

    sigma_value = 0.01  # variance of y
    Q = sigma_value*sp.eye(2)  # covariance matrix for alpha

    # generate alpha according to equation 5
    true_alpha = np.zeros((10,2,1))
    true_alpha[0] = a0
    for i in range(1,10):
        true_alpha[i] = F.dot(true_alpha[i-1])
    true_means = [Z.dot(a)[0].A1[0] for a in true_alpha]

    # generate observed alpha and y
    alpha = np.zeros((10,2,1))
    alpha[0] = a0
    for i in range(1,10):
        alpha[i] = F.dot(alpha[i-1]) + np.random.multivariate_normal([0,0], Q).reshape((-1,1))
    mu = [Z.dot(a)[0].A1[0] for a in alpha]
    y = np.random.multivariate_normal(mu, sigma_value*sp.eye(10), size=(1,200)).transpose()
    raw_means = y.mean(axis=1)

    rows = []
    for i,group in enumerate(y):
        for entry in group:
            rows.append([i, "A"] + entry.tolist())
    panel_series = PanelSeries.from_list(rows)

    # fit means with RSK
    rsk = RSK(F,Z)
    fitted_means = rsk.fit(panel_series, a0, Q, Q, sigma=sigma_value*sp.eye(1))
    smoothed_fitted_means = rsk.fit(panel_series, a0, Q, Q, smooth=True, sigma=sigma_value*sp.eye(1))
    return true_means, raw_means.flatten(), fitted_means.flatten(), smoothed_fitted_means.flatten(), y

def example():
    '''
    This example generates synthetic data following the model in equations 5-9 of
    http://folk.uio.no/jlind/papers/DP333.pdf
    '''
    true_means, raw_means, fitted_means, smoothed_fitted_means,y = trial()
    #
    # plot figure
    #
    fig = plt.figure()
    t=[i+1 for i in range(10)]
    plt.scatter(np.repeat(sp.matrix(t), 200, axis=1).A1, y.reshape((-1,)), linewidths=0.01, s=12, c="r", label="data")
    plt.scatter(jitter(t), true_means, linewidths=0.25, s=72, c="yellow",label="true mean")
    plt.scatter(jitter(t), fitted_means, linewidths=0.25, s=72, c="blue", label="rsk mean")
    plt.scatter(jitter(t), smoothed_fitted_means, linewidths=0.25, s=64, c="plum", label="rsk smoothed mean")
    plt.scatter(jitter(t), raw_means, linewidths=0.25, s=72, c="cyan", label="Naive mean")
    plt.legend()
    plt.title("Repeated Surveys Kalman Filter Demo")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()

def simulated_error(N):
    raw_errors = []
    rsk_errors = []
    smth_errors = []
    for i in range(N):
        true_means, raw_means, fitted_means, smoothed_fitted_means, _ = trial()
        raw_errors.append(np.sqrt(np.sum(np.square(true_means-raw_means))))
        rsk_errors.append(np.sqrt(np.sum(np.square(true_means-fitted_means))))
        smth_errors.append(np.sqrt(np.sum(np.square(true_means-smoothed_fitted_means))))
    print("Simulation results with %d trials:" % N)
    print("\tAvg. Naive Means Error: %.4f" % np.mean(raw_errors))
    print("\tAvg. RSK Error: %.4f" % np.mean(rsk_errors))
    print("\tAvg. Smoothed RSK Error: %.4f" % np.mean(smth_errors))

    # plot continuous density
    fig = plt.figure()
    density_raw = gaussian_kde(raw_errors)
    density_rsk = gaussian_kde(rsk_errors)
    xs = np.linspace(0,3,200)
    plt.plot(xs, density_raw(xs), color="blue", lw=3, label="Naive means error")
    plt.plot(xs, density_rsk(xs), color="red", lw=3, label="RSK means error")
    plt.legend()
    plt.title("Error Analysis with %d trials" % N)
    plt.xlabel("l2 error size")
    plt.ylabel("value")
    plt.show()

if __name__ == "__main__":
    example()
    simulated_error(1000)