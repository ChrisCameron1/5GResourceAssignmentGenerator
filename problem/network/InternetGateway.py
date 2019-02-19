__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

class InternetGateway():

    def __init__(self,
                name: str,
                bandwidth: float = 100.0
                 ):

        self.name = name
        self.bandwidth = bandwidth

        if not bandwidth:
            raise Exception("Bandwidth cannot be None for InternetGateway")

    def to_json(self):

        return {
            'name': self.name,
            'bandwidth': self.bandwidth
        }

    @staticmethod
    def from_json(obj):

        return InternetGateway(name=obj['name'],
                               bandwidth=obj['bandwidth'])


    @staticmethod
    def get(bandwidth=None):
        return InternetGateway(name='', bandwidth=bandwidth)
