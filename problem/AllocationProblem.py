__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from problem.network.Network import Network
from problem.traffic.Traffic import Traffic
from problem.slices.Slice import Slice
from problem.Options import Options

class AllocationProblem():

    def __init__(self,
                network: Network,
                slices: list(),
                traffic: Traffic,
                options: Options
                 ):
        '''

        :param network: Network layout and resources
        :param slices: List of slices
        :param traffic: Traffic that network needs to service. Representing number of users (datarate) on various slices
        :param options: Optimization options
        '''

        self.network = network
        self.slices = slices
        self.traffic = traffic
        self.options = options

    def to_json(self):

        return {
            'network': self.network.to_json(),
            'slices': [slice.to_json() for slice in self.slices],
            'traffic': self.traffic.to_json(),
            'options': self.options.to_json()
        }


    @staticmethod
    def from_json(obj):

        return AllocationProblem(network=Network.from_json(obj['network']),
                                 slices=[Slice.from_json(slice) for slice in obj['slices']],
                                 traffic=Traffic.from_json(obj['traffic']),
                                 options=Options.from_json(obj['options']))