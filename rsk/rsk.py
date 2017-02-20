import warnings
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
        for i in range(n_periods-1, 0, -1):
            B[i] = V_filter[i-1].dot(t(self.transition_matrix)).dot(inv(V[i]))
            alpha_smooth[i-1] = alpha_filter[i-1] + B[i].dot(alpha_smooth[i] - alpha[i])
            V_smooth[i-1] = V_filter[i-1] + B[i].dot(V_smooth[i]-V[i]).dot(t(B[i]))

        return alpha_smooth, V_smooth, B

    def fit(self, panel_series, a0, Q0, Q, smooth=True, sigma=None):
        '''
        Fit the RSK model to survey data
        :param panel_series: A PanelSeries object containing the survey data
        :param a0: array(n_alpha) initial value for the latent vector alpha
        :param Q0: array(n_alpha, n_alpha) Q0
        :param Q: array(n_alpha, n_alpha) Q
        :param smooth: boolean: apply the the smoothing algorithm
        :param sigma: specify a constant covariance matrix structure
        :return: alpha, alpha_filter, alpha_smooth, V, V_filter, V_smooth
        '''
        n_periods, n_vars = len(panel_series.times), panel_series.n_variables
        alpha, alpha_filter, alpha_smooth, _, _ , _,_ = self._fit(panel_series, a0, Q0, Q, smooth, sigma)

        # use smoothed values to make predictions?
        if smooth:
            alpha_pred = alpha_smooth
        else:
            alpha_pred = alpha_filter
        fitted_means = []
        for i in range(1, n_periods+1):
            n_groups = panel_series.group_counts_mask[i-1].shape[0]
            fitted_means.append(self.translation_matrix.dot(alpha_pred[i]).reshape(n_groups, n_vars))
        return fitted_means

    def _fit(self, panel_series, a0, Q0, Q, smooth=True, sigma=None):
        '''
        Fit the RSK model to survey data
        :param panel_series: A PanelSeries object containing the survey data
        :param a0: array(n_alpha) initial value for the latent vector alpha
        :param Q0: array(n_alpha, n_alpha) Q0
        :param Q: array(n_alpha, n_alpha) Q
        :param smooth: boolean: apply the the smoothing algorithm
        :param sigma: specify a constant covariance matrix structure
        :return: array(n_periods, n_vars) RSK estimated means
        '''

        # computations over the raw data
        n_periods, n_vars = len(panel_series.times), panel_series.n_variables
        y_means = panel_series.means()
        y_cov = panel_series.cov()

        # alpha hidden layer setup
        a0 = a0.reshape(-1,1)
        n_alpha = len(a0)
        alpha = sp.zeros((n_periods+1,  n_alpha, 1))
        alpha_filter = sp.zeros((n_periods+1, n_alpha, 1))
        alpha_filter[0] = a0

        # V covariance setup
        V = sp.zeros((n_periods + 1, n_alpha, n_alpha))
        V_filter = sp.zeros((n_periods+1, n_alpha, n_alpha))
        V_filter[0] = Q0

        # filter iterations
        transition_matrix, translation_matrix = self.transition_matrix, self.translation_matrix
        for i in range(1, n_periods+1):
            # compute group structure/covariance product

            if sigma is None:
                # no sigma provided, we use the covariance in the time slice
                _sigma = y_cov[i-1]
            elif len(sigma.shape) == 3:
                # sigma is varying in time, pin it to the current time slice
                _sigma = sigma[i-1]
            else:
                # constant sigma specified
                _sigma = sigma
            ng_sigma_inv = sp.kron(panel_series.group_counts_mask[i-1], inv(_sigma))

            # predict
            alpha[i] = transition_matrix.dot(alpha_filter[i-1, :])
            V[i] = transition_matrix.dot(V_filter[i-1, :]).dot(t(transition_matrix)) + Q
            V_filter[i] = inv(inv(V[i]) + t(translation_matrix).dot(ng_sigma_inv).dot(translation_matrix))
            alpha_filter[i] = alpha[i] + V_filter[i].dot(t(translation_matrix)).dot(ng_sigma_inv).dot(y_means[i - 1].reshape(-1,1) - translation_matrix.dot(alpha[i]))

        if smooth:
            alpha_smooth, V_smooth, smoothing_matrix = self.smooth(alpha, alpha_filter, V, V_filter)
        else:
            alpha_smooth, V_smooth, smoothing_matrix = None, None, None

        return alpha, alpha_filter, alpha_smooth, V, V_filter, V_smooth, smoothing_matrix

    def fit_em(self, panel_series, a0, Q0, sigma0=None, constant_sigma=False, tolerance=1e-4, max_iters=100):
        '''
        Fit the RSK model to survey data
        :param panel_series: A PanelSeries object containing the survey data
        :param a0: array(n_alpha) initial value for the latent vector alpha
        :param Q0: array(n_alpha, n_alpha) Q0
        :param Q: array(n_alpha, n_alpha) Q
        :param sigma0: initial covariance structure. If none, the covariance of the panel_series is used
        :param constant_sigma: boolean: if true, average sigma across time slices at end of each iteration
        :param fit_a0: boolean: if true, alpha0 is estimated during each iteration. (not recommended)
        :param tolerance: float
        :param max_iters: int
        :return: array(n_periods, n_vars) RSK estimated means
        '''

        n_periods, n_vars, n_alpha = len(panel_series.times), panel_series.n_variables, len(a0)
        Z,F = self.translation_matrix, self.transition_matrix
        Q = Q0
        sigma = sigma0
        error = 100 + tolerance
        alpha_stale = None
        iters = 0

        while error>tolerance and iters<max_iters:
            # fit alpha[0],...,alpha[T] the current values of the hyper parameters
            _, _, alpha_smooth, _, _, V_smooth, B = self._fit(panel_series, a0, Q0, Q, smooth=True, sigma=sigma)
            alpha = alpha_smooth
            V = V_smooth

            # update sigma
            sigma = sp.zeros((n_periods, n_vars, n_vars))
            for k, (time_label, panel) in enumerate(panel_series.data):
                n_groups = panel_series.group_counts_mask[k].shape[0]
                Nt = panel_series.group_counts_mask[k].sum()
                mu = Z.dot(alpha[k]).reshape((n_groups, n_vars))  # mu is a stacked vector
                ZVZ = Z.dot(V[k]).dot(t(Z))
                panel_mean = panel.mean()
                for j, group in enumerate(panel.data):
                    Ng = group.size
                    diff1 = (group.mean() - mu[j]).reshape((-1,1))
                    diff2 = (panel_mean - mu[j]).reshape((-1,1))
                    idx = (j * n_vars, (j + 1) * n_vars)
                    sigma[k] += (Ng / Nt) * (group.cov(bias=True) + diff1.dot(t(diff2)) + ZVZ[idx[0]:idx[1], idx[0]:idx[1]])

            if constant_sigma:
                sigma = sigma.mean(axis=0)

            # update Q
            Q = sp.zeros((n_alpha, n_alpha))
            N = sum([panel.size() for _,panel in panel_series.data])
            for k in range(n_periods):
                Nt = panel_series.group_counts_mask[k].sum()
                diff = alpha[k+1] - F.dot(alpha[k])
                Q += (Nt/N) * (diff.dot(t(diff)) + V[k+1] + F.dot(V[k]).dot(t(F)) - F.dot(B[k+1]).dot(V[k+1]) - V[k+1].dot(B[k+1]).dot(t(F)))

            # update a0, Q0
            Q0 = V[0]
            a0 = alpha[0]

            # compute the error and prepare for next step
            if alpha_stale is None:
                error = tolerance + 1000
            else:
                error = np.linalg.norm(alpha.reshape((-1,)) - alpha_stale.reshape((-1,)))
                print("Iteration %d, error: %.8f" % (iters,error))
            alpha_stale = alpha
            iters += 1

        if iters==max_iters:
            warnings.warn("RSK EM algorithm failed to converge before max iterations %d reached.  Current error=%.8f" % (max_iters, error))
        else:
            print("Converged in %d iterations..." % iters)
        return self.fit(panel_series, a0, Q0, Q, smooth=True, sigma=sigma), sigma
