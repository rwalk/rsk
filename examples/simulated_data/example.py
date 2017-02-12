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
    '''
    This example generates synthetic data following the model in equations 5-9 of
    http://folk.uio.no/jlind/papers/DP333.pdf
    '''
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
    fitted_means, sigma_fitted = rsk.fit_em(panel_series, a0, sigma_value*Q)
    return true_means, raw_means, fitted_means, y

def example():
    '''
    Run and plot a trial
    '''
    true_means, raw_means, fitted_means, y = trial()
    #
    # plot figure
    #
    fig = plt.figure()
    t = [i+1 for i in range(10)]
    plt.scatter(np.repeat(sp.matrix(t), 200, axis=1).A1, y.reshape((-1,)), linewidths=0.01, s=12, c="r", label="data")
    plt.scatter(jitter(t), true_means, linewidths=0.25, s=72, c="yellow",label="true mean")
    plt.scatter(jitter(t), fitted_means, linewidths=0.25, s=72, c="blue", label="rsk mean")
    plt.scatter(jitter(t), raw_means, linewidths=0.25, s=72, c="cyan", label="naive mean")
    plt.legend()
    plt.title("Repeated Surveys Kalman Filter Demo")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.savefig(os.path.join(os.path.dirname(__file__), "example.png"))
    plt.show()

def compute_error(X,Y):
    s = 0
    for x,y in zip(X,Y):
        s+=(x-y)**2
    return sp.sqrt(s)

def simulated_error(N):
    '''
    Repeats the simulation N times and plots the error density
    :param N:
    :return:
    '''
    raw_errors = []
    rsk_errors = []
    for i in range(N):
        true_means, raw_means, fitted_means, _ = trial()
        raw_errors.append(compute_error(true_means, raw_means))
        rsk_errors.append(compute_error(true_means, fitted_means))

    naive_mean, naive_std = np.mean(raw_errors), np.std(raw_errors)
    rsk_mean, rsk_std = np.mean(rsk_errors), np.std(rsk_errors)
    text = r"RSK: $\bar{\mu}=%.4f$, $\bar{\sigma}=%.4f$" % (rsk_mean, rsk_std) + "\n" + \
           r"Naive: $\bar{\mu}=%.4f$, $\bar{\sigma}=%.4f$" % (naive_mean, naive_std)


    print("Simulation results with %d trials:" % N)
    print("\tAvg. Naive means error=%.4f, stdev=%.4f" % (naive_mean,naive_std))
    print("\tAvg. RSK means error=%.4f, stdev=%.4f" % (rsk_mean,naive_std))

    # plot continuous density
    fig = plt.figure()
    density_raw = gaussian_kde(sp.vstack(raw_errors).flatten().tolist())
    density_rsk = gaussian_kde(sp.vstack(rsk_errors).flatten().tolist())

    xs = np.linspace(0,3,200)
    plt.plot(xs, density_raw(xs), color="blue", lw=3, label="Naive means error")
    plt.plot(xs, density_rsk(xs), color="red", lw=3, label="RSK means error")
    plt.legend()
    plt.title("Error density with %d trials" % N)
    plt.xlabel("l2 error size")
    plt.ylabel("value")
    plt.text(1.5, 0.6, text, bbox={'facecolor':'white', 'pad':10}, fontsize=12)
    plt.savefig(os.path.join(os.path.dirname(__file__), "error_density.png"))
    plt.show()


if __name__ == "__main__":
    example()
    simulated_error(100)