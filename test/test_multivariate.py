from unittest import TestCase
from util.oxcsv import parse_ox_csv
from rsk.rsk import RSK
import scipy as sp
from scipy.linalg import block_diag
from rsk.panel import PanelSeries
import numpy as np
import os.path
import json

class TestMultivariate(TestCase):
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/testdata/")

    def test_simple_mv(self):
        '''
        Test a simple multivariate example (to the best of our abilities given we don't have a ground truth for this
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

        rows = []
        for i,group in enumerate(y):
            for entry in group:
                rows.append([i, "A"] + entry.tolist())
        panel_series = PanelSeries.from_list(rows)

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

        fitted_means = rsk_filter.fit(panel_series, a0, Q0, Q, sigma=sigma )

        # check that all means are equal
        for row in fitted_means:
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

        rows = []
        for i,group in enumerate(y):
            for entry in group:
                rows.append([i, "A"] + entry.tolist())
        panel_series = PanelSeries.from_list(rows)

        alpha = sp.matrix(parse_ox_csv(os.path.join(self.datapath, "3/alpha.csv")),dtype=np.float64)[:,1:]
        ox_means = sp.array(parse_ox_csv(os.path.join(self.datapath, "3/means.csv")), dtype=np.float64).transpose()[1:]
        py_means,py_cov = panel_series.means(), panel_series.cov()

        # check means
        assert sp.allclose(ox_means, sp.vstack(py_means)), "Python means do not match OX means"

        # check alphas
        rsk_filter = RSK(sp.matrix(params["transition_matrix"]), sp.matrix(params["translation_matrix"]))
        rsk_alpha, alpha_filter, alpha_smooth, V, V_filter, V_smooth,_ = rsk_filter._fit(panel_series, sp.matrix(params["a0"]), sp.matrix(params["Q0"]),
                       sp.matrix(params["Q"]), sigma=sp.matrix(params["sigma"]))

        a1 = alpha.transpose()
        a2 = np.squeeze(rsk_alpha)[1:]
        assert sp.allclose(a1, a2), "Alpha does not match OX alpha"