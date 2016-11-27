import scipy as sp
from scipy import transpose as t
from scipy.linalg import inv

def rsk_filter(ymeans,
               transition_matrix,
               translation_matrix,
               group_count_matrix,
               sigma,
               a0,
               Q0,
               Q,
               n_groups
               ):

    n_periods, n_vars = ymeans.shape

    # alpha hidden layer setup
    n_alpha = len(a0)
    alpha = sp.zeros((n_periods+1,  n_alpha))
    alpha_filter = sp.zeros((n_periods+1, n_alpha))
    alpha_filter[0] = a0

    # V covariance setup
    V = sp.zeros((n_periods + 1, n_alpha, n_alpha))
    V_filter = sp.zeros((n_periods+1, n_alpha, n_alpha))
    V_filter[0]= Q0

    # group count/variance factor
    n_sigma_inv = n_groups*inv(sigma)

    for i in range(1, n_periods+1):
        # predict
        alpha[i] = transition_matrix.dot(alpha_filter[i-1, :])
        V[i] = transition_matrix.dot(V_filter[i-1, :]).dot(t(transition_matrix)) + Q

        # update
        V_filter[i] = inv(sp.linalg.inv(V[i]) + t(translation_matrix).dot(n_sigma_inv).dot(translation_matrix))
        alpha_filter[i] = alpha[i] + V_filter[i].dot(t(translation_matrix)).dot(n_sigma_inv).dot(ymeans[i - 1] - translation_matrix.dot(alpha[i]))

    return alpha, V, alpha_filter, V_filter


