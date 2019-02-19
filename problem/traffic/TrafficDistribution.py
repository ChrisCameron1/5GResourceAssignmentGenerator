__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

class TrafficDistribution():

    '''
    Data rate is approximately proportional to number of clients but there may be assymetric data per client.
    Is this discreteness of individual resources avoidable?
    '''

    def __init__(self,
                 slices=None
                 ):

        self.independent_groups_of_slices = slices

    def get_sample(self):
        return

    def get_samples(self, num_samples=100):

        return

    def to_json(self):

        return {
            'slice_loads': [(slice_load[0].to_json(), slice_load[1]) for slice_load in self.slice_loads]
        }
