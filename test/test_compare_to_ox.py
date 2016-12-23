from unittest import TestCase
from util.oxcsv import parse_ox_csv
from rsk.rsk import RSK
from rsk.panel import PanelSeries
import scipy as sp
import numpy as np
import os.path
import json

class TestCompareToOx(TestCase):
    '''This class compares our implementation against the Ox reference implementation'''
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/testdata/")

    def test1(self):
        '''
        Compare test case 1 to OX
        :return:
        '''
        with open(os.path.join(self.datapath, "1/params.json")) as f:
            params = json.loads(f.read())
        yy = sp.array(parse_ox_csv(os.path.join(self.datapath, "1/raw_data.csv")), dtype=np.float64).transpose()
        y = sp.reshape(yy, (101, 10, 1))[1:]
        alpha = sp.matrix(parse_ox_csv(os.path.join(self.datapath, "1/alpha.csv")),dtype=np.float64)
        ox_means = sp.array(parse_ox_csv(os.path.join(self.datapath, "1/means.csv")), dtype=np.float64).transpose()[1:]

        # ox cov is population cov but we want sample!
        ox_cov = (10.0/9.0)*sp.array(parse_ox_csv(os.path.join(self.datapath, "1/cov.csv")), dtype=np.float64).transpose()[1:]

        rows = []
        for i,group in enumerate(y):
            for entry in group:
                rows.append([i, "A"] + entry.tolist())
        panel_series = PanelSeries.from_list(rows)

        py_means, py_cov = panel_series.means(), panel_series.cov()

        # check means
        assert sp.allclose(ox_means, sp.vstack(py_means)), "Python means do not match OX means"

        # check covs
        assert sp.allclose(ox_cov, sp.vstack(py_cov)), "Python covariance does not match OX covariance."

        rsk_filter = RSK(sp.matrix(params["transition_matrix"]), sp.matrix(params["translation_matrix"]))
        rsk_filter.fit(panel_series, sp.matrix(params["a0"]), sp.matrix(params["Q0"]), sp.matrix(params["Q"]), sigma=sp.eye(1))

        # check alphas
        assert sp.allclose(alpha.A1.tolist()[1:], rsk_filter.alpha.flatten().tolist())


    def test2(self):
        '''
        Compare test case 2 to OX
        :return:
        '''
        with open(os.path.join(self.datapath, "2/params.json")) as f:
            params = json.loads(f.read())
        yy = sp.array(parse_ox_csv(os.path.join(self.datapath, "2/raw_data.csv")), dtype=np.float64).transpose()
        y = sp.reshape(yy, (101, 25, 1))[1:]

        # read the ox data
        alpha = sp.matrix(parse_ox_csv(os.path.join(self.datapath, "2/alpha.csv")),dtype=np.float64)
        ox_means = sp.array(parse_ox_csv(os.path.join(self.datapath, "2/means.csv")), dtype=np.float64).transpose()[1:]
        ox_cov = (25.0/24.0)*sp.array(parse_ox_csv(os.path.join(self.datapath, "2/cov.csv")), dtype=np.float64).transpose()

        rows = []
        for i,group in enumerate(y):
            for entry in group:
                rows.append([i, "A"] + entry.tolist())
        panel_series = PanelSeries.from_list(rows)
        py_means, py_cov = panel_series.means(), sp.vstack(panel_series.cov())

        # check means
        assert sp.allclose(ox_means, sp.vstack(py_means)), "Python means do not match OX means"

        # check covs
        assert sp.allclose(ox_cov[1:], py_cov), "Python covariance does not match OX covariance."

        rsk_filter = RSK(sp.matrix(params["transition_matrix"]), sp.matrix(params["translation_matrix"]))
        rsk_filter.fit(panel_series, sp.matrix(params["a0"]), sp.matrix(params["Q0"]), sp.matrix(params["Q"]), sigma=sp.eye(1))

        # check alphas
        assert sp.allclose(alpha.transpose()[1:].tolist(), rsk_filter.alpha.reshape(-1,6).tolist())

