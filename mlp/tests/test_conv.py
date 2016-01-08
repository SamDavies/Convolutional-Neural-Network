from unittest import TestCase

import numpy

from mlp.conv import ConvLinear


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
        conv = ConvLinear(1, 1,
                          image_shape=(28, 28),
                          kernel_shape=(5, 5),
                          stride=(1, 1),
                          irange=0.2,
                          rng=None,
                          conv_fwd=None,
                          conv_bck=None,
                          conv_grad=None)

        conv.W = numpy.zeros(conv.W.shape, dtype=numpy.float32)
        conv.W = numpy.ones((5, 5), dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        input = numpy.zeros(784, dtype=numpy.float32)
        input[0] = 1.0
        input[783] = 1.0
        output = numpy.zeros(conv.odim, dtype=numpy.float32)
        self.assertEqual(conv.fprop(input), output)
