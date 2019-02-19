__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

import sys
from problem.network.Rack import Rack
from problem.network.RemoteRadioHead import RemoteRadioHead
from problem.network.InternetGateway import InternetGateway


class DataCentre():

    def __init__(self,
                 name: str,
                 racks: list,
                 inter_rack_delay: float,
                 remote_radio_heads: list,
                 internet_gateway: InternetGateway,
                 type: str
                 ):
        '''
        Hardware contained within DataCentre
        :param name: Name of DataCentre
        :param racks: List of (Rack, copies) tuples
        :param inter_rack_delay: delay between racks
        :param internet_gateway: Internet Gateway function
        :param remote_radio_heads: List of (RadioHead, copies) tuples
        :param type: data centre type in {CORE, CRAN}
        '''

        self.name = name
        self.racks = racks
        self.inter_rack_delay = inter_rack_delay
        self.remote_radio_heads = remote_radio_heads
        self.internet_gateway = internet_gateway
        self.type = type

    def min_phy_capacity(self):

        min_capacity = sys.maxsize
        for rack in self.racks:
            for phy_processor in rack.phy_processors:
                if phy_processor.resources['CPU'] < min_capacity:
                    min_capacity = phy_processor.resources['CPU']

        return min_capacity

    def max_server_delay(self):
        max_server_delay = 0
        for rack in self.racks:
            if rack.inter_server_delay > max_server_delay:
                max_server_delay = rack.inter_server_delay
        return max_server_delay

    def to_json(self):
        return {
            'name': self.name,
            'racks': [r.to_json() for r in self.racks],  # TODO: Do the same thing that was done with servers in Rack
            'inter_rack_delay': self.inter_rack_delay,
            'radio_heads': [r.to_json() for r in self.remote_radio_heads],
            'internet_gateway': self.internet_gateway.to_json(),
            'type': self.type
        }

    @staticmethod
    def from_json(obj):
        racks = []
        radio_heads = []
        for r in obj['racks']:
            racks.append(Rack.from_json(r))
        for r in obj['radio_heads']:
            radio_heads.append(RemoteRadioHead.from_json(r))
        internet_gateway = InternetGateway.from_json(obj['internet_gateway'])

        return DataCentre(name=obj['name'],
                          racks=racks,
                          inter_rack_delay=obj['inter_rack_delay'],
                          remote_radio_heads=radio_heads,
                          internet_gateway=internet_gateway,
                          type=obj['type'])
