from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal

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

    def test_sigmoid_forward_propagation(self):
        """ Ensure that a sigmoid layer forwards propagates correctly """
        # Given
        rng = numpy.random.RandomState([2015, 10, 10])
        rng_state = rng.get_state()
        input_layer = numpy.asarray([-20.1, 52.4, 0, 0.05, 0.05, 49])
        output_layer = numpy.asarray([-20.1, 52.4, 0, 0.05, 0.05, 49, 20, 20])
        rng.set_state(rng_state)
        sigmoid = Sigmoid(idim=input_layer.shape[0], odim=output_layer.shape[0], rng=rng)

        # When
        forward = sigmoid.fprop(input_layer)
        expected = numpy.asarray([0.067, 0.728, 0.999, 0.512, 0.159, 0.584, 0.238, 0.932])

        # Then
        assert_array_almost_equal(forward, expected, decimal=3)

    def test_sigmoid_prime_large_positive(self):
        """ Ensure that a large positive value gives the correct sigmoid prime value """
        actual = Sigmoid.sigmoid_prime(20.0)
        self.assertAlmostEqual(0.0, actual)

    def test_sigmoid_prime_large_negative(self):
        """ Ensure that a large negative value gives the correct sigmoid prime value """
        actual = Sigmoid.sigmoid_prime(-20.0)
        self.assertAlmostEqual(0.0, actual)

    def test_sigmoid_prime_zero(self):
        """ Ensure that a zero value gives the correct sigmoid prime value """
        actual = Sigmoid.sigmoid_prime(0.0)
        self.assertAlmostEqual(0.25, actual)

    def test_sigmoid_backward_propagation(self):
        """ Ensure that a sigmoid layer forwards propagates correctly """
        # Given
        rng = numpy.random.RandomState([2015, 10, 10])
        rng_state = rng.get_state()
        input_layer = numpy.asarray([-20.1, 52.4, 0, 0.05, 0.05, 49])
        output_layer = numpy.asarray([-20.1, 52.4, 0, 0.05, 0.05, 49, 20, 20])
        rng.set_state(rng_state)
        sigmoid = Sigmoid(idim=input_layer.shape[0], odim=output_layer.shape[0], rng=rng)
        forward = sigmoid.fprop(input_layer)

        # When
        deltas, ograds = sigmoid.bprop(h=forward, igrads=output_layer)
        expected = numpy.asarray([1.518, -0.093, -0.703, 0.532, 1.96, 1.397])

        # Then
        assert_array_almost_equal(ograds, expected, decimal=3)
