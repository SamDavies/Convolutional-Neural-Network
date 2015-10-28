from unittest import TestCase

import math


def get_unit_count(layer_count):
    """
    For a given layer count, calculates the number of units per layer
    such that the total number of weights remains the same
    b = layer_count
    (784 * x) + ((b-1)*(x**2)) + (x * 10) = 635200
    ((b-1)*(x**2)) + (794 * x) - 635200 = 0
    """
    return (-794 + math.sqrt(794**2 - 4*(layer_count-1)*(-635200))) / (2 * (layer_count-1))
