__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

class RemoteRadioHead():

    '''
    Each resource within is parameterized by a load that will be proportional to the number of users
    '''

    def __init__(self,
                name: str,
                bandwidth: float
                 ):

        self.name = name
        self.bandwidth = bandwidth

    def get_copy(self, name=None):

        return RemoteRadioHead(name=name, bandwidth=self.bandwidth)

    def to_json(self):
        return {
            "name": self.name,
            "bandwidth": self.bandwidth,
        }

    @staticmethod
    def from_json(obj):

        name = obj['name']
        bandwidth = obj['bandwidth']

        return RemoteRadioHead(name=name, bandwidth=bandwidth)

