__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"

import pandas as pd
import json


class Traffic():

    def __init__(self,
                 traffic: pd.DataFrame  # DataFrame mapping traffic to slice, RRH pairs (str,float)),
                 ):
        self.traffic = traffic

    def to_json(self):
        return {
            'traffic': json.loads(self.traffic.to_json())
        }

    @staticmethod
    def from_json(obj):
        traffic = pd.DataFrame(obj['traffic'])
        return Traffic(traffic=traffic)
