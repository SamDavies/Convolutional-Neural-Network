from unittest import TestCase

import numpy

from mlp.conv import ConvLinear
from numpy.testing import assert_array_equal


class FeatureMapTestCase(TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState([2015, 10, 10])
        self.rng_state = self.rng.get_state()

    def test_create_linear(self):
        conv = ConvLinear(1,
                          image_shape=(28, 28),
                          kernel_shape=(5, 5),
                          stride=(1, 1),
                          irange=0.2,
                          rng=None,
                          conv_fwd=None,
                          conv_bck=None,
                          conv_grad=None)
        weights = conv.W[0]
        self.assertEqual(weights.shape, (24, 24, 5, 5))

    def test_fprop(self):
        """ Ensure that 1 forward prop pass works """
        conv = ConvLinear(1,
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
        input = numpy.zeros((28, 28), dtype=numpy.float32)
        input[0][0] = 1.0
        input[27][27] = 1.0

        weights = conv.W[0]
        num_rows_units = len(weights[0])
        num_cols_units = len(weights[1])

        expected = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0] = 1.0
        expected[23][23] = 1.0
        actual = conv.fprop_single_image(input, 0).reshape(24, 24)
        assert_array_equal(actual, expected)
