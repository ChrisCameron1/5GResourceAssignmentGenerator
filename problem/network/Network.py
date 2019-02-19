__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from problem.network.DataCentre import DataCentre
import sys

class Network:

    def __init__(self,
                data_centres: list,
                inter_data_centre_bandwidth: (),#Tuple[list, Any]
                inter_data_centre_delay: (),#Tuple[list, Any]
                 ):
        '''
        A network consists of a set of datacentres and inter/intra delays between servers within/across datacentres
        :param data_centres: List of data centres
        :param inter_data_centre_bandwidth: Pairwise bandwidth between each data centre
        :param inter_data_centre_delay: Pairwise delay between each data centre
        '''

        self.data_centres = data_centres #TODO: check that data_centre names are all unique
        self.inter_data_centre_bandwidth = inter_data_centre_bandwidth
        self.inter_data_centre_delay = inter_data_centre_delay
        self.check_data_centre_names_unique()

    def check_data_centre_names_unique(self):
        names = []
        for data_centre in self.data_centres:
            names.append(data_centre.name)

        if len(names) != len(set(names)):
            raise Exception('Data centre names:%s are not unique', names)

    def max_server(self, resource='CPU'):
        max_resource = 0
        for data_centre in self.data_centres:
            for rack in data_centre.racks:
                for server in rack.servers:
                    if server.resources[resource] > max_resource:
                        max_resource = server.resources[resource]

        return max_resource

    def max_data_centre_delay(self):
        return self.inter_data_centre_delay[1]

    def max_rack_delay(self):
        max_rack_delay = 0
        for data_centre in self.data_centres:
            inter_rack_delay = data_centre.inter_rack_delay
            if inter_rack_delay > max_rack_delay:
                max_rack_delay = inter_rack_delay
        return max_rack_delay

    def max_server_delay(self):
        max_server_delay = 0
        for data_centre in self.data_centres:
            max_data_centre_server_delay = data_centre.max_server_delay()
            if max_data_centre_server_delay > max_server_delay:
                max_server_delay = max_data_centre_server_delay

        return max_server_delay

    def min_phy_capacity(self):

        min_capacity = sys.maxsize
        for data_centre in self.data_centres:
            min_phy_capacity_data_centre = data_centre.min_phy_capacity()
            if min_phy_capacity_data_centre < min_capacity:
                min_capacity = min_phy_capacity_data_centre

        return min_capacity

    def to_json(self):
        # jsonStr= '{' +
        #         ""
        return {
            "DataCentres":[d.to_json() for d in self.data_centres],
            "InterDataCentreBandwidth":list(self.inter_data_centre_bandwidth),# Any combination of lists, tuple, dicts should just work
            "InterDataCentreDelay":list(self.inter_data_centre_delay)
        }

    @staticmethod
    def from_json(obj):
        data_centres = []
        for data_centre in obj['DataCentres']:
            data_centres.append(DataCentre.from_json(data_centre))

        inter_data_centre_bandwidth = obj['InterDataCentreBandwidth']
        inter_data_centre_delay = obj['InterDataCentreDelay']

        return Network(data_centres=data_centres,
                       inter_data_centre_bandwidth=inter_data_centre_bandwidth,
                       inter_data_centre_delay=inter_data_centre_delay)

    def get_capacity(self, resource='CPU'):

        capacity = 0.0
        for data_centre in self.data_centres:
            for rack in data_centre.racks:
                for server in rack.servers:
                    for resource_name in server.resources:
                        if resource == resource_name:
                            capacity += server.resources[resource_name]

        return capacity

