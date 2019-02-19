__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

class Options():


    def __init__(self,
                 objective_function: str,
                 min_cpu_per_server: float,
                 min_qos: float
                 ):

        self.objective_function = objective_function
        self.min_cpu_per_server = min_cpu_per_server
        self.min_qos = min_qos

    def to_json(self):

        return {
            'objective_function': self.objective_function,
            'min_cpu_per_server': self.min_cpu_per_server,
            'min_qos': self.min_qos
        }

    @staticmethod
    def from_json(obj):

        return Options(objective_function=obj['objective_function'],
                       min_cpu_per_server=float(obj['min_cpu_per_server']),
                       min_qos=float(obj['min_qos']))