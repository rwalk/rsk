from unittest import TestCase
from rsk.panel import *


class TestPanels(TestCase):
    '''This class tests the Panel data structures needed to support the RSK model'''
    # datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/testdata/")

    def test_groups(self):
        '''
        Test the group structure
        :return:
        '''
        data = [[1,1],[-1,2],[8,9]]
        group = Group("Western Territory of Endor", data)

        assert sp.allclose(group.mean(), [  2.66666667,  4.0])

        # note: variance is sample variance, which is presumably
        # the right measure here.
        assert sp.allclose(group.var(),  [ 22.33333333, 19.0])

        # group level covariance
        assert sp.allclose(group.cov(), sp.matrix([[22.33333333, 19.5],[19.5, 19.0]]))


    def test_panel(self):

        data_a = [[1,1],[-1,2],[8,9]]
        data_b = [[2,1],[0,23],[5,-19]]
        group_a = Group("Western Territory of Endor", data_a)
        group_b = Group("Eastern Territory of Endor", data_b)

        panel = Panel(42, [group_a, group_b])
        assert sp.allclose(panel.means(), sp.matrix([[ 2.33333333, 1.66666667], [2.66666667,  4.0 ]]))
        assert sp.allclose(panel.cov(), sp.matrix([[11.5,-12.9], [-12.9, 185.7667]]))

    def test_panel_series(self):

        # build up the panel.  Note we intentionally do this in reverse order
        data_2a = [[2,0],[0,0],[7,10]]
        data_2b = [[1,0],[0,22],[4,-17]]
        group_2a = Group("Western Territory of Endor", data_2a)
        group_2b = Group("Eastern Territory of Endor", data_2b)

        panel2 = Panel(1, [group_2a, group_2b])

        data_1a = [[1,1],[-1,2],[8,9]]
        data_1b = [[2,1],[0,23],[5,-19]]
        group_1a = Group("Western Territory of Endor", data_1a)
        group_1b = Group("Eastern Territory of Endor", data_1b)

        panel1 = Panel(0, [group_1a, group_1b])
        panel_series = PanelSeries([panel1, panel2], variable_names=["year","district", "ewok_count", "rebel_count"])
        assert sp.allclose(panel_series.group_count_mask[0], sp.diag([3,3]))
        assert sp.allclose(panel_series.group_count_mask[1], sp.diag([3,3]))

        covs = panel_series.cov()
        means = panel_series.means()
        assert sp.allclose(means[0], sp.matrix([[ 2.33333333, 1.66666667], [2.66666667,  4.0 ]]))
        assert sp.allclose(covs[0], sp.matrix([[11.5,-12.9], [-12.9, 185.7667]]))

    def test_panel_builders(self):
        data = [
            ["0", "Eastern Territory of Endor", 2, 1],
            ["0", "Eastern Territory of Endor", 0, 23],
            ["0", "Eastern Territory of Endor", 5, -19],
            ["0", "Western Territory of Endor", 1, 1],
            ["0", "Western Territory of Endor", -1, 2],
            ["0", "Western Territory of Endor", 8,9],
            ["1", "Eastern Territory of Endor", 1,0],
            ["1", "Eastern Territory of Endor", 0, 22],
            ["1", "Eastern Territory of Endor", 4, -17],
            ["1", "Western Territory of Endor", 2,0],
            ["1", "Western Territory of Endor", 0,0],
            ["1", "Western Territory of Endor", 7,10]
        ]

        # test list builder
        panel_series = PanelSeries.from_list(data, ["time", "region", "ewoks", "rebels"])
        panel_series.cov()
        covs = panel_series.cov()
        means = panel_series.means()
        assert sp.allclose(means[0], sp.matrix([[ 2.33333333, 1.66666667], [2.66666667,  4.0 ]]))
        assert sp.allclose(covs[0], sp.matrix([[11.5,-12.9], [-12.9, 185.7667]]))

        # test csv builder
        import csv
        with open("/tmp/panel-test.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "region", "ewoks", "rebels"])
            writer.writerows(data)

        panel_series = PanelSeries.from_csv("/tmp/panel-test.csv", 0,1)
        panel_series.cov()
        covs = panel_series.cov()
        means = panel_series.means()
        assert sp.allclose(means[0], sp.matrix([[ 2.33333333, 1.66666667], [2.66666667,  4.0 ]]))
        assert sp.allclose(covs[0], sp.matrix([[11.5,-12.9], [-12.9, 185.7667]]))






