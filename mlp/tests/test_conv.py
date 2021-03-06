from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal

from mlp.conv import ConvLinear
from mlp.costs import CECost
from mlp.dataset import MNISTDataProvider
from mlp.layers import MLP, Linear, Softmax, Sigmoid
from mlp.optimisers import SGDOptimiser
from mlp.schedulers import LearningRateFixed


class FeatureMapTestCase(TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState([2015, 10, 10])
        self.rng_state = self.rng.get_state()

    def test_create_linear(self):
        conv = ConvLinear(1, 2, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2)
        weights = conv.W
        self.assertEqual(weights.shape, (1, 2, 5, 5))

    def test_fprop_for_1_iamge(self):
        """ Ensure that 1 forward prop pass works for 1 image """
        conv = ConvLinear(1, 1, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2)
        conv.W = numpy.ones(conv.W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 1 feature map
        feature_maps = numpy.array([image])

        # make a batch of size 2
        batch = numpy.array([feature_maps])

        num_rows_units = conv.num_rows_units
        num_cols_units = conv.num_cols_units

        expected = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0] = 1.0
        expected[23][23] = 1.0
        actual = conv.fprop(batch)
        assert_array_equal(actual[0][0], expected)

    def test_fprop(self):
        """ Ensure that 1 forward prop pass works """
        conv = ConvLinear(1, 1, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2)
        conv.W = numpy.ones(conv.W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 1 feature map
        feature_maps = numpy.array([image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        num_rows_units = conv.num_rows_units
        num_cols_units = conv.num_cols_units

        expected = numpy.zeros((2, 1, num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0][0][0] = 1.0
        expected[0][0][23][23] = 1.0
        expected[1][0][0][0] = 1.0
        expected[1][0][23][23] = 1.0
        actual = conv.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)

    def test_fprop_from_2d_layer(self):
        """ Ensure that 1 forward prop pass works """
        conv = ConvLinear(1, 1, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2)
        conv.W = numpy.ones(conv.W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make a batch of size 2
        batch = numpy.array([image, image])

        num_rows_units = conv.num_rows_units
        num_cols_units = conv.num_cols_units

        expected = numpy.zeros((2, 1, num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0][0][0] = 1.0
        expected[0][0][23][23] = 1.0
        expected[1][0][0][0] = 1.0
        expected[1][0][23][23] = 1.0
        actual = conv.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)

    def test_fprop_many_out_feature_maps(self):
        """ Ensure that 1 forward prop pass works with many out feature maps """
        conv = ConvLinear(1, 2, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2)
        conv.W = numpy.ones(conv.W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 2 feature maps
        feature_maps = numpy.array([image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        num_rows_units = conv.num_rows_units
        num_cols_units = conv.num_cols_units

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

    def test_fprop_many_inp_feature_maps(self):
        """ Ensure that 1 forward prop pass works with many out feature maps """
        conv = ConvLinear(2, 1, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2)

        conv.W = numpy.ones(conv.W.shape, dtype=numpy.float32)

        # make an image of zeros with 1 in 2 corners
        image = numpy.zeros((784), dtype=numpy.float32)
        image[0] = 1.0
        image[783] = 1.0

        # make 2 feature maps
        feature_maps = numpy.array([image, image])

        # make a batch of size 2
        batch = numpy.array([feature_maps, feature_maps])

        num_rows_units = conv.num_rows_units
        num_cols_units = conv.num_cols_units

        expected = numpy.zeros((2, 1, num_rows_units, num_cols_units), dtype=numpy.float32)
        expected[0][0][0][0] = 2.0
        expected[0][0][23][23] = 2.0
        expected[1][0][0][0] = 2.0
        expected[1][0][23][23] = 2.0
        actual = conv.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)


class ConvLinearTestCase(TestCase):
    def test_model_fprop(self):
        """ Ensure that 1 forward prop pass works """
        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(ConvLinear(1, 1, image_shape=(28, 28), kernel_shape=(5, 5),stride=(1, 1),irange=0.2))
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
        model.add_layer(ConvLinear(1, 2, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Linear(idim=1152, odim=2))

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

        expected = numpy.array([[4., 4.], [4., 4.]])
        actual = model.fprop(batch)
        assert_array_equal(actual, expected, verbose=True)

    def test_model_bprop(self):
        """ Ensure that back prop works with pgrads """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=3, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=3, randomize=True)
        test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=3, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(ConvLinear(1, 1, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Softmax(idim=576, odim=10))

        lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=2)
        optimiser = SGDOptimiser(lr_scheduler=lr_scheduler, dp_scheduler=None,
                                 l1_weight=0.0, l2_weight=0.0)

        optimiser.train(model, train_dp, valid_iterator=valid_dp)

        tst_cost, tst_accuracy = optimiser.validate(model, test_dp)

        self.assertAlmostEqual(tst_accuracy, 0.0766, delta=0.005)

    def test_model_bprop_not_first_layer(self):
        """ Ensure that back prop works when the conv layer has 1 below it """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=3, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=3, randomize=True)
        test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=3, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=784))
        model.add_layer(ConvLinear(1, 1,image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Softmax(idim=576, odim=10))

        lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=2)
        optimiser = SGDOptimiser(lr_scheduler=lr_scheduler, dp_scheduler=None,
                                 l1_weight=0.0, l2_weight=0.0)

        optimiser.train(model, train_dp, valid_iterator=valid_dp)

        tst_cost, tst_accuracy = optimiser.validate(model, test_dp)

        self.assertAlmostEqual(tst_accuracy, 0.0966, delta=0.005)

    def test_model_bprop_3_feature_maps(self):
        """ Ensure that back prop works when the conv layer has 1 below it """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=1, randomize=True)
        test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=1, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=784))
        model.add_layer(ConvLinear(1, 3, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Softmax(idim=1728, odim=10))

        lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=2)
        optimiser = SGDOptimiser(lr_scheduler=lr_scheduler, dp_scheduler=None, l1_weight=0.0, l2_weight=0.0)

        optimiser.train(model, train_dp, valid_iterator=valid_dp)

        tst_cost, tst_accuracy = optimiser.validate(model, test_dp)

        self.assertAlmostEqual(tst_accuracy, 0.14, delta=0.005)

    def test_model_bprop_3_feature_maps_2_layers(self):
        """ Ensure that back prop works when the conv layer has 1 below it """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=1, randomize=True)
        test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=1, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=784))
        model.add_layer(ConvLinear(1, 3, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(ConvLinear(3, 3, image_shape=(24, 24), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Softmax(idim=1200, odim=10))

        lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=2)
        optimiser = SGDOptimiser(lr_scheduler=lr_scheduler, dp_scheduler=None, l1_weight=0.0, l2_weight=0.0)

        optimiser.train(model, train_dp, valid_iterator=valid_dp)

        tst_cost, tst_accuracy = optimiser.validate(model, test_dp)

        self.assertAlmostEqual(tst_accuracy, 0.12, delta=0.005)





