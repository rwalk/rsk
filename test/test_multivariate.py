from unittest import TestCase
from util.oxcsv import parse_ox_csv
from rsk.rsk import RSK
import scipy as sp
from scipy.linalg import block_diag
from scipy import transpose as t
import numpy as np
import os.path
import json

class TestMultivariate(TestCase):
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/testdata/")

    def test_simple_mv(self):
        '''
        Test a simple multivariate example (to the best of our abillities given we don't have a ground truth for this
        example).
        :return:
        '''

        # load the second OX reference data set.
        with open(os.path.join(self.datapath, "2/params.json")) as f:
            params = json.loads(f.read())
        yy = sp.array(parse_ox_csv(os.path.join(self.datapath, "2/raw_data.csv")), dtype=np.float64).transpose()
        y = sp.reshape(yy, (101, 25, 1))[1:]

        # now we'll make a few copies of the reference data and stack them along the third axis.
        y = sp.squeeze(sp.stack((y,y,y,y,y), axis=2))

        # run the filtering
        Z = sp.matrix(params["translation_matrix"])
        F = sp.matrix(params["transition_matrix"])
        a0 = sp.matrix(params["a0"]).reshape(-1,1)
        Q0 = sp.matrix(params["Q0"])
        Q = sp.matrix(params["Q"])
        sigma = sp.eye(5)

        F = block_diag(F,F,F,F,F)
        Z = block_diag(Z,Z,Z,Z,Z)

        a0 = sp.vstack((a0,a0,a0,a0,a0))
        Q0 = block_diag(Q0,Q0,Q0,Q0,Q0)
        Q = block_diag(Q,Q,Q,Q,Q)

        rsk_filter = RSK(F,Z)

        fitted_means = rsk_filter.fit(y,sigma , a0, Q0, Q )[1:]

        # check that all means are equal
        for row in fitted_means.tolist():
            assert np.allclose(row[0:2], row[2:4]), "Measurements differ unexpectedly."


    def test_compare_ox_multi(self):
        '''
        Compare against multivariate ox reference implementation
        :return:
        '''
        with open(os.path.join(self.datapath, "3/params.json")) as f:
            params = json.loads(f.read())
        yy = sp.array(parse_ox_csv(os.path.join(self.datapath, "3/raw_data.csv")), dtype=np.float64)

        # unstack yy
        subarrays = [yy[i*15:i*15+15,] for i in range(10)]
        y = sp.stack(tuple(subarrays), axis=0)


        alpha = sp.matrix(parse_ox_csv(os.path.join(self.datapath, "3/alpha.csv")),dtype=np.float64)[:,1:]
        ox_means = sp.array(parse_ox_csv(os.path.join(self.datapath, "3/means.csv")), dtype=np.float64).transpose()[1:]
        ox_cov = sp.array(parse_ox_csv(os.path.join(self.datapath, "3/cov.csv")), dtype=np.float64).transpose()
        py_means,py_cov = RSK.aggregate_raw_data(y)

        # check means
        assert sp.allclose(ox_means, py_means), "Python means do not match OX means"

        # check covs
        assert sp.allclose(ox_cov[1:], py_cov), "Python covariance does not match OX covariance."

        # check alphas
        rsk_filter = RSK(sp.matrix(params["transition_matrix"]), sp.matrix(params["translation_matrix"]))
        rsk_filter.fit(y, sp.matrix(params["sigma"]), sp.matrix(params["a0"]), sp.matrix(params["Q0"]), sp.matrix(params["Q"]))

        a1 = alpha.transpose()
        a2 = np.squeeze(rsk_filter.alpha)
        assert sp.allclose(a1, a2), "Alpha does not match OX alpha"