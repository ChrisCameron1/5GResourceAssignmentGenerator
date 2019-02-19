__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from json import JSONEncoder

class Resource(JSONEncoder):


    '''
    Each resource within is parameterized by a load that will be proportional to the number of users
    '''

    VALID_RESOURCE_TYPES = ['CPU', 'MEMORY', 'IO', 'MEM']

    def __init__(self,
                name: str,
                quantity_per_load: float
                 ):
        '''
        Resources of a server
        :param name:
        :param quantity:
        '''
        self.check_valid_resource_type(name)

        self.name = name
        self.quantity_per_load = quantity_per_load

    def check_valid_resource_type(self, name):

        if not name.upper() in Resource.VALID_RESOURCE_TYPES:
            raise Exception('Resource type: %s not recognized' % (name))

    def is_within(self,
                  function_proportion_pairs: list):
        '''

        :param resources:
        :return:
        '''

        total_resource = 0.0
        for (function,proportion) in function_proportion_pairs:
            total_resource += function.resources[self.name]

        return total_resource <= self.quantity_per_load

    def default(self):

        return self.to_json()

    def to_json(self):

        return {
            "resource_name": self.name,
            "quantity_per_load": self.quantity_per_load
        }


    @staticmethod
    def from_json(obj):

        return Resource(name=obj['resource_name'], quantity_per_load=obj['quantity_per_load'])

