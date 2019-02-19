__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from problem.network.Server import Server
from problem.network.PhyProcessor import PhyProcessor

class Rack():


    def __init__(self,
                 name: str,
                 servers: list,
                 phy_processors: list,
                 inter_server_delay: float):
        '''

        :param servers: List of (Server, quantity) tuples that lie within DataCentre
        :param inter_server_delay:
        '''

        self.name = name
        self.servers = servers
        self.phy_processors = phy_processors
        self.inter_server_delay = inter_server_delay

        self.check_data_centre_names_unique()
        self.check_phy_processors_subset()

    def check_data_centre_names_unique(self):
        names = []
        for server in self.servers:
            names.append(server.name)

        if len(names) != len(set(names)):
            raise Exception('Server names:%s are not unique within rack: %s', names, self.name)

    def check_phy_processors_subset(self):
        server_names = [server.name for server in self.servers]
        phy_processor_names = [phy_processor.name for phy_processor in self.phy_processors]

        if not set(phy_processor_names).issubset(server_names):
            raise Exception("Phy processors are not strict subset of servers!")

    def get_copy(self, name=None):

        return Rack(name=name,
                    servers=self.servers,
                    phy_processors=self.phy_processors,
                    inter_server_delay=self.inter_server_delay)

    def to_json(self):
        return {
            "name": self.name,
            "servers": [s.to_json() for s in self.servers],
            "phy_processors": [p.to_json() for p in self.phy_processors],
            "inter_server_delay": self.inter_server_delay,
        }


    @staticmethod
    def from_json(obj):

        servers = []
        for server in obj['servers']:
            # number = server['number']
            # server.pip('nmber',None)
            servers.append(Server.from_json(server))

        phy_processors = []
        for phy_processor in obj['phy_processors']:
            phy_processors.append(PhyProcessor.from_json(phy_processor))

        inter_server_delay = obj['inter_server_delay']
        name = obj['name']

        return Rack(name=name,
                    servers=servers,
                    phy_processors=phy_processors,
                    inter_server_delay=inter_server_delay)
