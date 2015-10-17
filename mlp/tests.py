from unittest import TestCase

from mlp.layers import Sigmoid

__author__ = 'Sam Davies'


class SigmoidTestCase(TestCase):
    def test_sigmoid_large_positive(self):
        """ Ensure that a large positive value gives the correct sigmoid value """
        actual = Sigmoid.sigmoid(20.0)
        self.assertAlmostEqual(1.0, actual)

    def test_sigmoid_large_negative(self):
        """ Ensure that a large negative value gives the correct sigmoid value """
        actual = Sigmoid.sigmoid(-20.0)
        self.assertAlmostEqual(0.0, actual)

    def test_sigmoid_zero(self):
        """ Ensure that a zero value gives the correct sigmoid value """
        actual = Sigmoid.sigmoid(0.0)
        self.assertAlmostEqual(0.5, actual)