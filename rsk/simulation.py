import scipy as sp
import numpy as np
from scipy.linalg import cholesky

def randn_matrix(n, m):
    return sp.matrix(sp.randn(n, m))

def generate_random_y(n_periods,
                      n_vars,
                      n_individuals,
                      n_alpha,
                      transition_matrix,
                      translation_matrix,
                      a0,
                      sigma,
                      Q,
                      Q0=None
                      ):
    '''
     The y is n_individuals x n_vars x n_periods array
      representing observations of n_individuals on n_vars measured
      at n_periods distinct points in time.

    :param n_periods:
    :param n_vars:
    :param n_individuals:
    :param n_alpha:
    :param transition_matrix:
    :param translation_matrix:
    :param a0:
    :param sigma:
    :param Q:
    :param Q0:
    :return:
    '''

    chol_sigma = cholesky(sigma).transpose()
    try:
        chol_Q = cholesky(Q).transpose()
    except np.linalg.linalg.LinAlgError:
        chol_Q = 0

    if Q0 is not None:
        chol_Q0 = cholesky(Q0).transpose()
    else:
        chol_Q0 = 0

    y = None

    alpha = randn_matrix(n_alpha, 1).dot(chol_Q0) + a0
    for t in range(n_periods):
        alpha = transition_matrix.dot(alpha) + randn_matrix(n_alpha, 1).dot(chol_Q)
        y_col = sp.array((sp.ones((n_individuals, 1)).dot(translation_matrix.dot(alpha))).transpose() + randn_matrix(n_vars, n_individuals).dot(chol_sigma), ndmin=3)
        if y is None:
            y = sp.array(y_col, ndmin=3)
        else:
            y = sp.vstack((y, y_col))
    return y


def random_walk_example(n_periods):
    # right now, we just generate the data according to a random walk.
    a0,Q0, Q = map(lambda x: sp.matrix([x]),[0,1,1])
    n_individuals = 10
    sigma = sp.eye(n_individuals)
    n_vars=1
    n_alpha = 1
    transition_matrix = sp.matrix([1])
    translation_matrix = sp.matrix([1])
    y = generate_random_y(n_periods, n_vars, n_individuals, n_alpha, transition_matrix, translation_matrix, a0, sigma, Q, Q0)
    return y

def arama22_example(n_periods):
    # right now, we just generate the data according the ARMA(2,2)
    n_alpha = 6
    n_vars = 5
    n_individuals = 3

    a0 = sp.zeros((n_alpha,1))
    Q0 = sp.eye(n_alpha)
    sigma = sp.eye(n_individuals)

    # Q here is a rank 1 matrix
    q = sp.matrix([1,0,0,0,0,0]).transpose()
    Q = q.dot(q.transpose())

    # column normalized transition matrix
    transition_matrix = sp.matrix([
        [0,0,0,0,0,0],
		[1,0,0.5,0.1,0.5,0.3],
		[0,1,0,0,0,0],
		[0,0,1,0,0,0],
		[1,0,0,0,0,0],
		[0,0,0,0,1,0]
    ])

    translation_matrix = sp.matrix([0,1,0,0,0,0])
    return generate_random_y(n_periods, n_vars, n_individuals, n_alpha, transition_matrix, translation_matrix, a0, sigma, Q)




if __name__ == "__main__":
    #print(random_walk_example(5).shape)
    print(arama22_example(7))