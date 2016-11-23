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
                      sigma
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
    :return:
    '''

    chol_sigma = cholesky(sigma).transpose()

    # preallocate y
    y = np.zeros((n_periods,  n_individuals, n_vars))

    alpha = randn_matrix(n_alpha, 1) + a0
    for t in range(n_periods):
        alpha = transition_matrix.dot(alpha) + randn_matrix(n_alpha, 1)
        y[t] = sp.ones((n_individuals, 1)).dot(translation_matrix.dot(alpha)) + (randn_matrix(n_vars, n_individuals).dot(chol_sigma)).transpose()
    return y


def random_walk_example(n_periods, n_individuals):
    # right now, we just generate the data according to a random walk.
    a0,Q0,Q = map(lambda x: sp.matrix([x]),[0,1,1])
    sigma = sp.eye(n_individuals)
    n_vars=1
    n_alpha = 1
    transition_matrix = sp.matrix([1])
    translation_matrix = sp.matrix([1])
    y = generate_random_y(n_periods, n_vars, n_individuals, n_alpha, transition_matrix, translation_matrix, a0, sigma)
    return y

def arama22_example(n_periods, n_individuals):
    # right now, we just generate the data according the ARMA(2,2)
    n_alpha = 6
    n_vars = 1

    a0 = sp.zeros((n_alpha,1))
    sigma = sp.eye(n_individuals)

    # Q here is a rank 1 matrix
    #q = sp.matrix([1,0,0,0,0,0]).transpose()
    #Q = q.dot(q.transpose())

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
    return generate_random_y(n_periods, n_vars, n_individuals, n_alpha, transition_matrix, translation_matrix, a0, sigma)

if __name__ == "__main__":
    print(random_walk_example(5, 10))
    print(arama22_example(2,5).shape)