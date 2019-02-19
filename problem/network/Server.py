__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

import random

import problem.slices.VirtualFunction


class Server():


    def __init__(self,
                name: str,
                resources: dict(),
                type: str
                 ):

        self.name = name
        self.resources = resources
        self.type = type

    def copy_server(self, name=None):

        return Server(name=name, resources=self.resources, type=self.type)

    def add_function(self,
                     virtual_function: problem.slices.VirtualFunction,
                     fraction: float):
        '''
        Observer pattern: When virtual function added, call to virtual function and update it's list of connected servers and amount of use
        :param function:
        :param fraction:
        :return:
        '''
        self.check_function_can_be_added()
        self.function_proportions.append( (virtual_function, fraction))



    def check_function_can_be_added(self,virtual_function):

        for resource in self.resources:
            if not resource.is_within(self.function_porportions + [virtual_function]):
                raise Exception('Virtual function %s can not be added to Server %s' % (virtual_function, self))

    def to_json(self):

        return {
            "name": self.name,
            "resources": self.resources,
            "type": self.type
        }


    @staticmethod
    def from_json(obj):
        name = obj['name']
        resources = obj['resources']
        type = obj['type']

        return Server(name=name, resources=resources, type=type)

    @staticmethod
    def get_random(server_cpu_settings: list() =None,
                              server_mem_settings: list() =None,
                              server_IO_settings: list()=None,
                              cpu_clock_speeds: list()=None):
        # TODO: not a range. Need to use random.choice
        resources = {"CPU": random.choice(server_cpu_settings),
                    "MEM": random.choice(server_mem_settings),
                    "IO": random.choice(server_IO_settings),
                    "Clock": random.choice(cpu_clock_speeds)
                    }
        type=""
        return Server(name="random", resources=resources, type=type)




