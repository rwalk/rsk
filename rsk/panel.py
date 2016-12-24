import csv
import scipy as sp

def is_numeric(entry):
    try:
        float(entry)
        return True
    except ValueError:
        return False

class PanelSeries:

    def __init__(self, panels, variable_names=None):
        '''
        A panel series is a time series of panels. Generally, it should not be constructed directly
        but rather from a builder method such as PanelSeries.from_csv
        :param panels:
        :param variable_names: names for the variables in the series.  The first two names are the names of time and group vars.
        '''

        # if the time index is numeric, need to convert to float so that data will be properly sorted
        if all([is_numeric(panel.time) for panel in panels]):
            for panel in panels:
                panel.time = float(panel.time)

        self.data = sorted([(panel.time, panel) for panel in panels], key=lambda x: x[0])
        self.times = [time for time,panel in self.data]
        self.variable_names = variable_names

        # verify that we have balanced individuals
        groups = [group.name for group in panels[0].data]
        if not all([groups == [group.name for group in panel.data] for panel in panels]):
            raise ValueError("Currently, all panels must have the same members in the same order.")
        self.groups = groups

        #
        # compute group masks and check variables
        #
        group_counts_mask = []
        _, n_vars = panels[0].data[0].data.shape
        self.n_variables = n_vars
        for panel in panels:
            group_sizes = [group.data.shape[0] for group in panel.data]
            var_counts = [group.data.shape[1] for group in panel.data]
            if not all([v==n_vars for v in var_counts]):
                raise ValueError("Must have same number of variables for each individual!")
            group_counts_mask.append(sp.diag(group_sizes))
        self.group_counts_mask = group_counts_mask

    def means(self):
        '''
        compute group means matrix (n_groups x n_vars)
        :return: list of matrices by (n_groups x n_vars)
        '''
        return [panel.means() for t,panel in self.data]

    def cov(self):
        '''
        compute covariance matrix for each time slice across all groups
        :return: list of matrices (n_vars x n_vars)
        '''
        return [panel.cov() for t,panel in self.data]

    @staticmethod
    def from_csv(filename, time_index, group_index, header=True, drop_missing=True):
        '''
        Load a data set from a csv file.
        :param filename:
        :param time_index: column position of the time index
        :param group_index: column position of the group index
        :param header: treat first row as header
        :return:
        '''
        rowlist = []
        skip_index = {time_index, group_index}
        colnames = None
        with open(filename) as f:
            reader = csv.reader(f)
            if header:
                colnames = next(reader)
                colnames = [colnames[time_index], colnames[group_index]] + [entry for i,entry in enumerate(colnames) if i not in skip_index]

            for row in reader:
                if drop_missing and not all([is_numeric(entry) for i,entry in enumerate(row) if i not in skip_index]):
                    continue

                #put the time and group indices at start of row, followed by data
                rowlist.append([row[time_index], row[group_index]] + [float(entry) for i,entry in enumerate(row) if i not in skip_index])
        return PanelSeries.from_list(rowlist, colnames)

    @staticmethod
    def from_list(rowlist, variable_names = None):
        '''
        Construct a panel series from a list of rows (in an arbitrary order).  Each
        row must contain a time index and a group index at index 0 and 1 respectively.
        :param rowlist:
        :param colnames:
        :return: PanelSeries
        '''

        N = len(rowlist[0])
        if N<3:
            raise ValueError("Row must contain time, group, and at least one observation.")

        # collect panels out of rows (which are in arbitrary order)
        panel_dict = {}
        for row in rowlist:
            if len(row)!=N:
                raise ValueError("All rows in rowlist must have the same length.")
            t,g, data = row[0], row[1], row[2:]

            if t not in panel_dict:
                panel_dict[t] = {}
            panel = panel_dict[t]

            if g not in panel:
                panel[g] = []
            panel[g].append(data)

        # process each panel and order by time index
        panels = []
        for t,panel in sorted(panel_dict.items(), key=lambda x:x[0]):
            groups = []
            for g,data in sorted(panel.items(), key=lambda x:x[0]):
                groups.append(Group(g, data))
            panels.append(Panel(t, groups))

        return PanelSeries(panels, variable_names)


class Group:

    def __init__(self, name, observations):
        '''
        A group is a collection of observations on a set of individuals at a specific moment in time.
        :param name: label of the group
        :param observations: list of rows of observations of individuals or an n_individuals by n_vars matrix
        '''
        self.name = name
        if type(observations) is list:
            self.data = sp.array(observations)
        else:
            self.data = observations

    def mean(self):
        return sp.mean(self.data, axis=0)

    def var(self):
        '''
        Group level variance
        :return:
        '''
        return sp.var(self.data, axis=0, ddof=1)

    def cov(self):
        '''
        Group level covariance
        :return: covariance matrix (n_vars x n_vars)
        '''
        return sp.cov(self.data, rowvar=False, ddof=1)

class Panel:

    def __init__(self, time, groups):
        '''
        A panel is a collection of groups obeserved at the same same moment in time.
        :param time: time at which the groups were observered
        :param groups: a list of groups
        '''
        self.time = time
        self.data = sorted(groups, key=lambda x: x.name)

    def means(self):
        return sp.vstack([group.mean() for group in self.data])

    def var(self):
        '''
        variance across all groups in this time slice
        :return:
        '''
        return sp.vstack([group.var() for group in self.data])

    def cov(self):
        '''
        covariance across all groups in this time slice
        :return: covariance matrix (n_vars x n_vars)
        '''
        M = sp.vstack([group.data for group in self.data])
        return sp.matrix(sp.cov(M, rowvar=False, ddof=1))