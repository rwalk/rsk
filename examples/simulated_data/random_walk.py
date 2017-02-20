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

# Example settings
SIGMA_SCALE = 100
Q_SCALE = 0.001
N_TIMES = 10
N_POINTS = 500
A0_SCALE = 0.01
Q0_SCALE = 10000
RHO = 1.01
TRUE_A0 = 1

def trial():
    '''
    This example generates synthetic data following the model in equations 5-9 of
    http://folk.uio.no/jlind/papers/DP333.pdf
    '''
    Z = sp.eye(1)
    F = RHO*sp.eye(1)
    a0 = np.matrix([TRUE_A0])
    Q = Q_SCALE*sp.eye(1)  # covariance matrix for alpha
    Q0 = Q0_SCALE * sp.eye(1)

    # generate alpha according to equation 5
    true_alpha = np.zeros((N_TIMES+1,1,1))
    true_alpha[0] = a0
    for i in range(1, N_TIMES+1):
        true_alpha[i] = F.dot(true_alpha[i-1])
    true_means = [Z.dot(a)[0] for a in true_alpha[1:]]

    # generate 'observed' alpha and y
    alpha = np.zeros((N_TIMES+1,1,1))
    alpha[0] = a0
    for i in range(1,N_TIMES+1):
        alpha[i] = F.dot(alpha[i-1]) + np.random.multivariate_normal([0], Q).reshape((-1,1))
    mu = [Z.dot(a)[0].tolist()[0] for a in alpha[1:]]
    y = np.random.multivariate_normal(mu, SIGMA_SCALE*sp.eye(N_TIMES), size=(1, 200)).transpose()
    raw_means = y.mean(axis=1)

    rows = []
    for i,group in enumerate(y):
        for entry in group:
            rows.append([i, "A"] + entry.tolist())
    panel_series = PanelSeries.from_list(rows)

    # fit means with RSK
    rsk = RSK(F,Z)
    fitted_means = rsk.fit(panel_series, sp.matrix([A0_SCALE]), Q0, Q)
    fitted_means_em, sigma = rsk.fit_em(panel_series, sp.matrix([A0_SCALE]), Q0, max_iters=100, tolerance=1e-6)
    return true_means, raw_means, fitted_means, fitted_means_em, y

def example():
    '''
    Run and plot a trial
    '''
    true_means, raw_means, fitted_means, fitted_means_em, y = trial()
    #
    # plot figure
    #
    focal_point_size = 72
    fig = plt.figure()
    t = [i+1 for i in range(N_TIMES)]
    plt.plot(t, true_means, linewidth=1.5,  c="black", label="true mean")
    plt.scatter(t, fitted_means, linewidths=0, s=focal_point_size, c="red", label="rsk mean")
    plt.scatter(t, fitted_means_em, linewidths=0, s=focal_point_size, c="blue", label="rsk mean")
    plt.scatter(t, raw_means, linewidths=0, s=focal_point_size, c="yellow", label="naive mean")
    plt.legend()
    plt.title("Repeated Surveys Kalman Filter Demo")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.savefig(os.path.join(os.path.dirname(__file__), "random_walk_example.png"))
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
    rsk_em_errors = []
    for i in range(N):
        true_means, raw_means, fitted_means, fitted_means_em, _ = trial()
        raw_errors.append(compute_error(true_means, raw_means))
        rsk_errors.append(compute_error(true_means, fitted_means))
        rsk_em_errors.append(compute_error(true_means, fitted_means_em))
    naive_mean, naive_std = np.mean(raw_errors), np.std(raw_errors)
    rsk_mean, rsk_std = np.mean(rsk_errors), np.std(rsk_errors)
    rsk_em_mean, rsk_em_std = np.mean(rsk_em_errors), np.std(rsk_em_errors)
    text = r"RSK: $\bar{\mu}=%.4f$, $\bar{\sigma}=%.4f$" % (rsk_mean, rsk_std) + "\n" + \
           r"RSK-EM: $\bar{\mu}=%.4f$, $\bar{\sigma}=%.4f$" % (rsk_em_mean, rsk_em_std) + "\n" + \
           r"Naive: $\bar{\mu}=%.4f$, $\bar{\sigma}=%.4f$" % (naive_mean, naive_std)

    print("Simulation results with %d trials:" % N)
    print("\tAvg. Naive means error=%.4f, stdev=%.4f" % (naive_mean, naive_std))
    print("\tAvg. RSK means error=%.4f, stdev=%.4f" % (rsk_mean, rsk_std))
    print("\tAvg. RSK EM means error=%.4f, stdev=%.4f" % (rsk_em_mean, rsk_em_std))

    # plot continuous density
    fig = plt.figure()
    density_raw = gaussian_kde(sp.vstack(raw_errors).flatten().tolist())
    density_rsk = gaussian_kde(sp.vstack(rsk_errors).flatten().tolist())
    density_rsk_em = gaussian_kde(sp.vstack(rsk_em_errors).flatten().tolist())

    xs = np.linspace(0,3, N)
    plt.plot(xs, density_raw(xs), color="yellow", lw=3, label="Naive means error")
    plt.plot(xs, density_rsk(xs), color="red", lw=3, label="RSK means error")
    plt.plot(xs, density_rsk_em(xs), color="blue", lw=3, label="RSK EM means error")
    plt.legend()
    plt.title("Error density with %d trials" % N)
    plt.xlabel("l2 error size")
    plt.ylabel("value")
    plt.text(1.5, 0.6, text, bbox={'facecolor':'white', 'pad':10}, fontsize=12)
    plt.savefig(os.path.join(os.path.dirname(__file__), "random_walk_error_density.png"))
    plt.show()

if __name__ == "__main__":
    example()
    simulated_error(100)