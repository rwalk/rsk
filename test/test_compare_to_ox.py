from unittest import TestCase
from util.oxcsv import parse_ox_csv
from rsk.rsk import RSK
import scipy as sp
import os.path
import json

class TestCompareToOx(TestCase):
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/testdata/")

    def test1(self):
        '''
        Compare test case 1 to OX
        :return:
        '''
        with open(os.path.join(self.datapath, "1/params.json")) as f:
            params = json.loads(f.read())
        yy = sp.array(parse_ox_csv(os.path.join(self.datapath, "1/raw_data.csv"))).transpose()
        y = sp.reshape(yy, (101, 10, 1))[2:]
        alpha = sp.matrix(parse_ox_csv(os.path.join(self.datapath, "1/alpha.csv"))).transpose()[2:]

        print("Y shape: %d x %d x %d" % y.shape)
        print("alpha shape: %d x %d" % alpha.shape)

        rsk_filter = RSK(sp.matrix(params["transition_matrix"]), sp.matrix(params["translation_matrix"]))
        rsk_filter.fit(y, sp.matrix(params["sigma"]), sp.matrix(params["a0"]), sp.matrix(params["Q0"]), sp.matrix(params["Q"]))

        for a1,a2 in zip(alpha, rsk_filter.alpha):
            print(a1,a2)
        assert sp.allclose(alpha, rsk_filter.alpha)


