__author__ = "Chris Cameron"
__email__ = "cchris13@cs.ubc.ca"
__version__ = "0.1"
__license__ = "GPL version 3"


class SliceGroup():

    def __init__(self,slices=None,process=None):

        self.slices = slices

    def max_difference(self, resource='CPU'):

        max_difference = 0
        for slice in self.slices:
            slice_difference = slice.max_difference(resource=resource)
            if slice_difference > max_difference:
                max_difference = slice_difference

        return max_difference


