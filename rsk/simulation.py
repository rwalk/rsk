import scipy as sp
from scipy.linalg import cholesky
from scipy import transpose as t, ones
from rsk import rsk_filter


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
    y = sp.zeros((n_periods,  n_individuals, n_vars))

    alpha = randn_matrix(n_alpha, 1) + a0
    for i in range(n_periods):
        alpha = transition_matrix.dot(alpha) + randn_matrix(n_alpha, 1)
        y[i] = translation_matrix.dot(alpha) + randn_matrix(n_individuals, n_vars)
    return y

def aggregate_raw_data(y):
    '''
    Compute means and covariances of the raw data y
    :param y: array(n_periods x n_individuals x n_vars)
    :return:
    '''
    m = sp.mean(y, axis=0)
    c = sp.var(y, axis=0)
    return m,c


def random_walk_example(n_periods, n_individuals):
    # right now, we just generate the data according to a random walk.
    a0,Q0,Q = map(lambda x: sp.matrix([x]),[0,1,1])
    n_vars=1
    sigma = sp.eye(n_vars)
    n_alpha = 1
    transition_matrix = sp.matrix([1])
    translation_matrix = sp.matrix([1])
    y = generate_random_y(n_periods, n_vars, n_individuals, n_alpha, transition_matrix, translation_matrix, a0, sigma)
    ymeans, ycov = aggregate_raw_data(y)
    result = rsk_filter(ymeans,
               translation_matrix,
               translation_matrix,
               sp.eye(1),
               sigma,
               a0,
               Q0,
               Q,
               1 # n groups
               )

    return result

def arama22_example(n_periods, n_individuals):
    # right now, we just generate the data according the ARMA(2,2)
    n_alpha = 6
    n_vars = 1

    a0 = sp.zeros((n_alpha,1))
    sigma = 1

    # Q here is a rank 1 matrix
    # q = sp.matrix([1,0,0,0,0,0]).transpose()
    # Q = q.dot(q.transpose())

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
    print(random_walk_example(10, 15))
    #print(arama22_example(2,5).shape)