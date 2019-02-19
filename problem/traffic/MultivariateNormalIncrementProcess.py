__author__ = "Chris Cameron, Rex Chen"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

import matplotlib
import json
import numpy as np
import pandas as pd
from problem.traffic.Traffic import Traffic
import random
from problem.traffic.TrafficDistribution import TrafficDistribution
from problem.slices import Slice

matplotlib.use('agg')


class MultivariateNormalIncrementProcess(TrafficDistribution):

    def __init__(self,
                 slices: list = None,
                 starting_count=None,
                 rates=None,
                 covariance_matrix=None,
                 time_intervals=1,
                 min_phy_traffic_capacity=8.0):
        """
        Build random process which is addition of multivariate normal distributions

        Slices must be in same order as means and covariance matrix
        :param slices:
        :param starting_count:
        :param rates:
        :param covariance_matrix:
        :param num_samples:
        :param time_intervals:
        """

        self.slices = slices
        self.starting_count = starting_count
        self.rates = rates
        self.covariance_matrix = covariance_matrix
        self.time_intervals = time_intervals
        self.min_phy_traffic_capacity = min_phy_traffic_capacity

        super(MultivariateNormalIncrementProcess, self).__init__(slices=slices)

    @staticmethod
    def parameterize(slices=None,
                     network=None,
                     time_intervals=None,
                     correlations=None,
                     proportions=None,
                     proportion_variance=None,
                     constraining_resource='CPU',
                     min_phy_traffic_capacity=8.0,
                     approx_network_load=0.8):

        print("Parameterizing slice traffic distributions...")

        resources_per_slice = {}
        for slice in slices:
            slice_name = slice.name
            resources_per_slice[slice_name] = slice.get_resource_requirements_rate(resource=constraining_resource)

        # Define amount of server to be taken up
        consumption_proportion = approx_network_load

        aggregate_traffic = {}
        cumulative_network_capacity = network.get_capacity(resource=constraining_resource)

        #print("Cumulative Network Capacity: %2f CPUs" % cumulative_network_capacity)

        slice_requirements = {}
        slice_resources_sum = 0
        for slice in slices:
            slice_name = slice.name
            # Resource requirements per bitrate proportion
            slice_resource_requirement = slice.get_resource_requirements_rate(resource=constraining_resource)
            #print("Slice resource requirement: %s: %s" % (slice_name, slice_resource_requirement))
            resource_requirements = slice_resource_requirement * proportions[slice_name] * slice.get_max_resource_requirements_rate(resource=constraining_resource)
            slice_requirements[slice_name] = resource_requirements
            slice_resources_sum += resource_requirements

        adjusted_traffic = (
                                   cumulative_network_capacity / slice_resources_sum) * consumption_proportion  # CPU / CPU/bitrate = cumulative bitrate
        #print("Total Bitrate: %f" % (adjusted_traffic))
        per_resource_traffic_count = {}
        for slice in slices:
            slice_name = slice.name
            # print(slice_name)

            # print(proportions)
            per_resource_traffic_count[slice_name] = adjusted_traffic * proportions[
                slice_name]  # TODO: We need to deal with proportion before computing traffic

        #print("Slice bitrate distribution: %s" % (per_resource_traffic_count))
        aggregate_traffic = per_resource_traffic_count

        starting_count = [aggregate_traffic[slice.name] for slice in slices]

        slice_names = []
        for slice in slices:
            slice_names.append(slice.name)

        covariance_matrix = pd.DataFrame(0.0, index=slice_names, columns=slice_names)

        # Set relationships
        for correlation in correlations:
            if correlation[0] == correlation[1]:
                raise Exception("Can't set diagonal with anything but 1")

            # Should ensure symmetry!
            covariance_matrix.loc[correlation[0], correlation[1]] = correlation[2]
            covariance_matrix.loc[correlation[1], correlation[0]] = correlation[2]

        if not proportion_variance:
            proportion_variance = {}
            for slice_index in range(len(slices)):
                slice = slices[slice_index]
                proportion_variance[slice.name] = 1 / starting_count[slice_index]

        # Set diagonals representing covariance within every slice
        for slice_index in range(len(slices)):
            slice = slices[slice_index]
            slice_name = slice.name
            covariance_matrix.loc[slice_name, slice_name] = starting_count[slice_index] * proportion_variance[
                slice_name]
        if min_phy_traffic_capacity == 0.0:
            raise Exception("Min Phy traffic capacity is 0.0. Not traffic can be allocated!")
        return MultivariateNormalIncrementProcess(slices=slices, starting_count=starting_count,
                                                  covariance_matrix=covariance_matrix, time_intervals=time_intervals,
                                                  min_phy_traffic_capacity=min_phy_traffic_capacity)

    def get_samples(self, num_samples=None, network=None):
        '''
        Get traffic snapshot for every time interval
        :param num_samples:
        :param time_interval:
        :return: list of Traffic Objects
        '''

        if num_samples < self.time_intervals:
            print(
                "Warning: Fewer samples requested than time intervals. Num samples will be set to %d." % self.time_intervals)
        num_samples_per_interval = max(1, int(
            num_samples / self.time_intervals))  # Require at least one sample for interval
        num_samples = int(num_samples_per_interval * self.time_intervals)
        #print('Num samples: %d' % num_samples)

        mean_on_off_distribution = [[0 for i in range(len(self.slices))] for j in range(self.time_intervals)]
        for i in range(self.time_intervals):
            next_i = 0 if i == self.time_intervals - 1 else i + 1

            for j in range(len(self.slices)):
                decrement_proportion = 0  # traffic_decrements[next_i] - traffic_decrements[i]
                mean_on_off_distribution[i][j] = self.starting_count[
                                                     j] * decrement_proportion / num_samples_per_interval  # TODO: Rates should be set in get_sample, not here!!

        index = range(num_samples)
        slice_names = []
        slice_RRHs = []
        for slice in self.slices:
            slice_names.append(slice.name)
            for data_centre in network.data_centres:
                for RRH in data_centre.remote_radio_heads:
                    slice_RRHs.append(slice.name + RRH.name)
        traffic_samples = pd.DataFrame(0, index=index, columns=slice_names + slice_RRHs, )
        for j in range(len(self.slices)):
            traffic_samples.loc[0, j] = self.starting_count[j]

        j = 0
        for slice in self.slices:
            traffic_samples.loc[0, slice.name] = self.starting_count[j]
            j += 1

        slice_counter = np.zeros(len(self.slices))
        div = num_samples_per_interval  # int(num_samples / self.time_intervals)

        num_RRHs = 0
        for data_centre in network.data_centres:
            for RRH in data_centre.remote_radio_heads:
                num_RRHs += 1

        samples = []

        # Initialize
        sorted_intervals = sorted([random.random() for i in range(num_RRHs + 1)])
        d = []
        for slice_index in range(len(slice_names)):
            #print("Slice:%s" % slice.name)
            slice = self.slices[slice_index]
            slice_name = slice.name
            j = 0
            for data_centre in network.data_centres:
                for RRH in data_centre.remote_radio_heads:
                    traffic_proportion = (sorted_intervals[j + 1] - sorted_intervals[j])
                    rrh_traffic = traffic_proportion * traffic_samples.loc[0][slice_name]
                    traffic_samples.loc[0, slice_name + RRH.name] = min(rrh_traffic, self.min_phy_traffic_capacity)
                    bitrate = traffic_samples.loc[0][slice_name + RRH.name]
                    if bitrate == 0.0:
                        raise Exception(
                            'Bitrate is 0 for sample %d, slice %s at RRH %s' % (0, slice_name, RRH.name))
                    d.append({'DataCentre': data_centre.name,
                              'RRH': RRH.name,
                              'Slice': slice_name,
                              'bitrate': bitrate})
                    j += 1
        vars = pd.DataFrame(d)
        samples.append(Traffic(vars))

        for i in range(self.time_intervals):
            multivariate_samples = np.random.multivariate_normal(mean_on_off_distribution[i], self.covariance_matrix,
                                                                 div)

            for sample in range(i * div, (1 + i) * div):
                d = []
                if sample == 0:
                    continue

                # Get duration of bursts from all new events
                for slice_index in range(len(slice_names)):
                    slice = self.slices[slice_index]
                    slice_name = slice.name
                    increment = round(multivariate_samples[sample - (i * div)][slice_index], 0)
                    traffic_samples.loc[sample][slice_name] = traffic_samples.loc[sample - 1][slice_name] + increment

                    sorted_intervals = sorted([random.random() for i in range(num_RRHs + 1)])
                    j = 0
                    for data_centre in network.data_centres:
                        for RRH in data_centre.remote_radio_heads:
                            traffic_proportion = (sorted_intervals[j + 1] - sorted_intervals[j])
                            rrh_increment = traffic_proportion * increment
                            updated_rrh_traffic = traffic_samples.loc[sample - 1][slice_name + RRH.name] + rrh_increment
                            traffic_samples.loc[sample, slice_name + RRH.name] = min(updated_rrh_traffic,
                                                                                     self.min_phy_traffic_capacity)
                            bitrate = traffic_samples.loc[sample][slice_name + RRH.name]
                            if bitrate == 0.0:
                                raise Exception(
                                    'Bitrate is 0 for sample %d, slice %s at RRH %s' % (sample, slice_name, RRH.name))
                            d.append({'DataCentre': data_centre.name,
                                      'RRH': RRH.name,
                                      'Slice': slice_name,
                                      'bitrate': bitrate})
                            j += 1

                    if traffic_samples.loc[sample][slice_name] < 0:
                        traffic_samples.loc[sample][slice_name] = 0

                    slice_counter[slice_index] += 1

                vars = pd.DataFrame(d)
                samples.append(Traffic(vars))

        return samples

    def to_json(self):
        values_dict = {'slices': self.slices,
                       'starting_count': self.starting_count,
                       'rates': self.rates,
                       'covariance_matrix': self.covariance_matrix.to_dict()}

        return values_dict

    @staticmethod
    def from_json(self, path=None):
        with open(path, 'r') as fp:
            values_dict = json.loads(fp.read())

        slices = values_dict['slices']
        starting_count = values_dict['starting_count']
        time_intervals = values_dict['time_intervals']
        min_phy_traffic_capacity = values_dict['min_phy_traffic_capacity']
        covariance_matrix = pd.DataFrame.from_dict(values_dict['covariance_matrix'])

        return MultivariateNormalIncrementProcess(slices=slices, starting_count=starting_count,
                                                  covariance_matrix=covariance_matrix, time_intervals=time_intervals,
                                                  min_phy_traffic_capacity=min_phy_traffic_capacity)
