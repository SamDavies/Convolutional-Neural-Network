from unittest import TestCase

import numpy

from mlp.conv import ConvLinear
from numpy.testing import assert_array_equal

from mlp.costs import CECost
from mlp.layers import MLP, Linear


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
        weights = conv.W[0]
        self.assertEqual(weights.shape, (24, 24, 5, 5))

    def test_fprop_for_1_iamge(self):
        """ Ensure that 1 forward prop pass works for 1 image """
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
        input = numpy.zeros((28, 28), dtype=numpy.float32)
        input[0][0] = 1.0
        input[27][27] = 1.0

        weights = conv.W[0]
        num_rows_units = len(weights[0])
        num_cols_units = len(weights[1])

        expected = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0] = 1.0
        expected[23][23] = 1.0
        actual = conv.fprop_single_feature_map([input], 0)
        assert_array_equal(actual, expected)

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
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 1 feature map
        feature_maps = numpy.array([image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        weights = conv.W[0]
        num_rows_units = len(weights[0])
        num_cols_units = len(weights[1])

        expected = numpy.zeros((2, 1, num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0][0][0] = 1.0
        expected[0][0][23][23] = 1.0
        expected[1][0][0][0] = 1.0
        expected[1][0][23][23] = 1.0
        actual = conv.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)

    def test_fprop_from_2d_layer(self):
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
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make a batch of size 2
        batch = numpy.array([image, image])

        weights = conv.W[0]
        num_rows_units = len(weights[0])
        num_cols_units = len(weights[1])

        expected = numpy.zeros((2, 1, num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0][0][0] = 1.0
        expected[0][0][23][23] = 1.0
        expected[1][0][0][0] = 1.0
        expected[1][0][23][23] = 1.0
        actual = conv.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)

    def test_fprop_many_out_feature_maps(self):
        """ Ensure that 1 forward prop pass works with many out feature maps """
        conv = ConvLinear(1, 2,
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
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 2 feature maps
        feature_maps = numpy.array([image, image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        weights = conv.W[0]
        num_rows_units = len(weights[0])
        num_cols_units = len(weights[1])

        expected = numpy.zeros((2, 2, num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0][0][0] = 1.0
        expected[0][0][23][23] = 1.0
        expected[1][0][0][0] = 1.0
        expected[1][0][23][23] = 1.0
        # copy the previous feature map
        expected[0][1] = expected[0][0]
        expected[1][1] = expected[1][0]
        actual = conv.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)


class ConvLinearTestCase(TestCase):
    def test_model_fprop(self):
        """ Ensure that 1 forward prop pass works """
        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(
                ConvLinear(1, 1,
                           image_shape=(28, 28),
                           kernel_shape=(5, 5),
                           stride=(1, 1),
                           irange=0.2,
                           rng=None,
                           conv_fwd=None,
                           conv_bck=None,
                           conv_grad=None)
        )
        model.add_layer(Linear(idim=576, odim=2))

        model.layers[0].W = numpy.ones(model.layers[0].W.shape, dtype=numpy.float32)
        model.layers[1].W = numpy.ones(model.layers[1].W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 1 feature map
        feature_maps = numpy.array([image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        expected = numpy.array([[2., 2.], [2., 2.]])
        actual = model.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)

    def test_model_fprop_multi_feature_maps(self):
        """ Ensure that 1 forward prop pass works """
        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(
                ConvLinear(1, 2,
                           image_shape=(28, 28),
                           kernel_shape=(5, 5),
                           stride=(1, 1),
                           irange=0.2,
                           rng=None,
                           conv_fwd=None,
                           conv_bck=None,
                           conv_grad=None)
        )
        model.add_layer(Linear(idim=1152, odim=2))

        model.layers[0].W = numpy.ones(model.layers[0].W.shape, dtype=numpy.float32)
        model.layers[1].W = numpy.ones(model.layers[1].W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 1 feature map
        feature_maps = numpy.array([image, image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        expected = numpy.array([[4., 4.], [4., 4.]])
        actual = model.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)
