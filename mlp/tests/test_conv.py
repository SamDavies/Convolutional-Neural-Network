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
        self.assertEqual(conv.get_weights().shape, (576, 5, 5))
        new = conv.get_weights()
        self.assertEqual(new[0][0][0], conv.W[0][0])
        self.assertEqual(new[0][4][4], conv.W[24][0])

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
        new[0][4][4] = 47.0
        self.assertEqual(new[0][4][4], conv.W[24][0])

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

        self.assertEqual(conv.W.shape, (25, 576))
