import scipy as sp
import numpy as np
from scipy import transpose as t
from scipy.linalg import inv

class RSK:

    def __init__(self, transition_matrix, translation_matrix):
        '''

        :param transition_matrix:  array(n_alpha, n_alpha) transition model for latent alpha vector
        :param translation_matrix: array(n_vars, n_alpha) translation vector mapping alpha to means
        :return:
        '''
        self.translation_matrix = translation_matrix
        self.transition_matrix = transition_matrix

    @staticmethod
    def aggregate_raw_data(y):
        '''
        Compute means and covariances of the raw data y across individuals at each point in time measurement.
        :param y: array(n_periods x n_individuals x n_vars)
        :return: array(n_periods, n_vars), array(n_periods, n_vars)
        '''
        #TODO: Accomadate non-trivial group structure
        m = sp.mean(y, axis=1).reshape(-1,y.shape[-1])        # coerce to a matrix, else we get shape errors.
        c = sp.var(y, axis=1).reshape(-1,y.shape[-1])
        return m,c


    def smooth(self, alpha, alpha_filter, V, V_filter):
        '''
        Backwards recursive smoother
        :param alpha:
        :param alpha_filter:
        :param V:
        :param V_filter:
        :return:
        '''
        n_periods, n_alpha, _ = V.shape
        alpha_smooth = sp.zeros(alpha.shape, )
        alpha_smooth[-1] = alpha_filter[-1]
        V_smooth = sp.zeros(V.shape, np.float64)
        V_smooth[-1] = V_filter[-1]
        B = sp.zeros(V.shape, np.float64)

        for i in range(n_periods-1,0,-1):
            B[i] = V_filter[i-1].dot(t(self.transition_matrix)).dot(inv(V[i]))
            alpha_smooth[i-1] = alpha_filter[i-1] + B[i].dot(alpha_smooth[i] - alpha[i])
            V_smooth[i-1] = V_filter[i-1] + B[i].dot(V_smooth[i]-V[i]).dot(t(B[i]))

        return alpha_smooth, V_smooth

    def fit(self, data, sigma, a0, Q0, Q, smooth=False):
        '''
        :param data: array(n_periods, n_individuals, n_vars) survey data
        :param sigma: array(n_vars, n_vars) empirical covariance of the n_vars measured variables
        :param a0: array(n_alpha) initial value for the latent vector alpha
        :param Q0: array(n_alpha, n_alpha) Q0
        :param Q: array(n_alpha, n_alpha) Q
        :return: array(n_periods, n_vars) RSK estimated means
        '''

        # computations over the raw data
        n_periods, n_individuals, n_vars = data.shape
        y_means, y_cov = self.aggregate_raw_data(data)

        # group structure encoding
        n_groups = sp.diag([data.shape[1]])         #TODO: for now, only one group containing everything
        n_sigma_inv = sp.kron(n_groups, inv(sigma)) #TODO: for now, non-dynamic

        # alpha hidden layer setup
        a0 = a0.reshape(-1,1)
        n_alpha = len(a0)
        alpha = sp.zeros((n_periods+1,  n_alpha, 1))
        alpha_filter = sp.zeros((n_periods+1, n_alpha, 1))
        alpha_filter[0] = a0

        # V covariance setup
        V = sp.zeros((n_periods + 1, n_alpha, n_alpha))
        V_filter = sp.zeros((n_periods+1, n_alpha, n_alpha))
        V_filter[0]= Q0

        # filter iterations
        transition_matrix, translation_matrix = self.transition_matrix, self.translation_matrix
        for i in range(1, n_periods+1):
            # predict
            alpha[i] = transition_matrix.dot(alpha_filter[i-1, :])
            V[i] = transition_matrix.dot(V_filter[i-1, :]).dot(t(transition_matrix)) + Q
            V_filter[i] = inv(inv(V[i]) + t(translation_matrix).dot(n_sigma_inv).dot(translation_matrix))
            alpha_filter[i] = alpha[i] + V_filter[i].dot(t(translation_matrix)).dot(n_sigma_inv).dot(y_means[i - 1].reshape(-1,1) - translation_matrix.dot(alpha[i]))

        if smooth:
            alpha, V = self.smooth(alpha, alpha_filter, V, V_filter)

        # remove the dummy NULL entry at start of arrays
        alpha, alpha_filter, V, V_filter = alpha[1:], alpha_filter[1:], V[1:], V_filter[1:]

        self.alpha = alpha
        self.alpha_filter = alpha_filter
        self.V=V
        self.V_filter = V_filter

        fitted_means = sp.zeros((n_periods, n_vars))
        for i in range(n_periods):
            fitted_means[i] = sp.squeeze(self.translation_matrix.dot(alpha[i]))

        return fitted_means
