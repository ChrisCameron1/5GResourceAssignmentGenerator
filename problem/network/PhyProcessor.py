__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from problem.network.Server import Server


class PhyProcessor(Server):

    def __init__(self,
                 name: str,
                 resources: dict(),
                 type: str
                 ):

        super(Server, self).__init__(name=name, resource=resources, type=type)




