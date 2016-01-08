from unittest import TestCase

import numpy

from mlp.conv import ConvLinear
from numpy.testing import assert_array_equal


class FeatureMapTestCase(TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState([2015, 10, 10])
        self.rng_state = self.rng.get_state()

    def test_create_linear(self):
        conv = ConvLinear(1, 1,
                          image_shape=(28, 28),
                          kernel_shape=(5, 5),
                          stride=(1, 1),
                          irange=0.2,
                          rng=None,
                          conv_fwd=None,
                          conv_bck=None,
                          conv_grad=None)
        self.assertEqual(conv.get_weights().shape, (24, 24, 5, 5))
        new = conv.get_weights()
        self.assertEqual(new[0][1][0][1], conv.W[1][1])
        self.assertEqual(new[0][1][4][4], conv.W[24][1])

    def test_change_reshaped_array(self):
        conv = ConvLinear(1, 1,
                          image_shape=(28, 28),
                          kernel_shape=(5, 5),
                          stride=(1, 1),
                          irange=0.2,
                          rng=None,
                          conv_fwd=None,
                          conv_bck=None,
                          conv_grad=None)
        new = conv.get_weights()
        new[0][0][4][4] = 47.0
        self.assertEqual(new[0][0][4][4], conv.W[24][0])

    def test_fprop(self):
        """ Ensure that 1 forward prop pass works """
        conv = ConvLinear(1, 1,
                          image_shape=(28, 28),
                          kernel_shape=(5, 5),
                          stride=(1, 1),
                          irange=0.2,
                          rng=None,
                          conv_fwd=None,
                          conv_bck=None,
                          conv_grad=None)

        conv.W = numpy.ones(conv.W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        input = numpy.zeros(784, dtype=numpy.float32)
        input[0] = 1.0
        input[783] = 1.0

        weights = conv.get_weights()
        num_rows_units = len(weights[0])
        num_cols_units = len(weights[1])

        expected = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0] = 1.0
        expected[23][23] = 1.0
        actual = conv.fprop(input).reshape(24, 24)
        assert_array_equal(actual, expected)
