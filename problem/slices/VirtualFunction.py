__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

from problem.network.Resource import Resource
from json import JSONEncoder

class VirtualFunction(JSONEncoder):

    def __init__(self,
                 resources: list(),
                 name:'',
                 phy_processor_constrained: bool = False,
                 data_centre_type: str = 'RAN'
                 ):

        self.resources = resources
        self.name = name
        self.phy_processor_constrained = phy_processor_constrained
        self.data_centre_type = data_centre_type

        self.resource_names = []

        if self.phy_processor_constrained and self.data_centre_type == 'CORE':
            raise Exception("Function constrained to PhyProcessors must not be on CORE network.")

        for resource in self.resources:
            self.resource_names.append(resource.name)

    def get_quantity_per_rate(self, resource):
        for function_resource in self.resources:
             if function_resource.name == resource:
                 return function_resource.quantity_per_load

        raise Exception("Function %s does require resource %s" % (self.name, resource))

    def default(self):

        return self.to_json()

    def to_json(self):

        return {
            'name': self.name,
            'resources': [r.to_json() for r in self.resources],
            'phy_processor_constrained': str(self.phy_processor_constrained),
            'data_centre_type': self.data_centre_type
        }

    @staticmethod
    def from_json(obj):
        resources = []
        for r in obj['resources']:
            resources.append(Resource.from_json(r))

        phy_processor_constrained = obj['phy_processor_constrained'] in ['True', 'true', 't', 'T']
        return VirtualFunction(name=obj['name'],
                               resources=resources,
                               phy_processor_constrained=phy_processor_constrained,
                               data_centre_type=obj['data_centre_type'])

    def __hash__(self):
        #TODO: Fix terrible hash function!!
        return hash(self.name)


