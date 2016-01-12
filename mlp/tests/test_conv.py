from unittest import TestCase

import numpy
import time

from mlp.conv import ConvLinear
from numpy.testing import assert_array_equal

from mlp.costs import CECost, MSECost
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

    def test_conv_linear_fprop(self):
        conv = ConvLinear(1, 2, image_shape=(4, 4), kernel_shape=(2, 2), stride=(1, 1), irange=0.2)
        out = self.conv_linear_fprop(conv)
        self.assertTrue(out)

    def conv_linear_fprop(self, layer, kernel_order='ioxy', kernels_first=True,
                           dtype=numpy.float):
        """
        Tests forward propagation method of a convolutional layer.

        Checks the outputs of `fprop` method for a fixed input against known
        reference values for the outputs and raises an AssertionError if
        the outputted values are not consistent with the reference values. If
        tests are all passed returns True.

        Parameters
        ----------
        layer : instance of Layer subclass
            Convolutional (linear only) layer implementation. It must implement
            the methods `get_params`, `set_params` and `fprop`.
        kernel_order : string
            Specifes dimension ordering assumed for convolutional kernels
            passed to `layer`. Default is `ioxy` which corresponds to:
                input channels, output channels, image x, image y
            The other option is 'oixy' which corresponds to
                output channels, input channels, image x, image y
            Any other value will raise a ValueError exception.
        kernels_first : boolean
            Specifies order in which parameters are passed to and returned from
            `get_params` and `set_params`. Default is True which corresponds
            to signatures of `get_params` and `set_params` being:
                kernels, biases = layer.get_params()
                layer.set_params([kernels, biases])
            If False this corresponds to signatures of `get_params` and
            `set_params` being:
                biases, kernels = layer.get_params()
                layer.set_params([biases, kernels])
        dtype : numpy data type
             Data type to use in numpy arrays passed to layer methods. Default
             is `numpy.float`.

        Raises
        ------
        AssertionError
            Raised if output of `layer.fprop` is inconsistent with reference
            values either in shape or values.
        ValueError
            Raised if `kernel_order` is not a valid order string.
        """
        inputs = numpy.arange(96).reshape((2, 3, 4, 4)).astype(dtype)
        kernels = numpy.arange(-12, 12).reshape((3, 2, 2, 2)).astype(dtype)
        if kernel_order == 'oixy':
            kernels = kernels.swapaxes(0, 1)
        elif kernel_order != 'ioxy':
            raise ValueError('kernel_order must be one of "ioxy" and "oixy"')
        biases = numpy.arange(2).astype(dtype)
        true_output = numpy.array(
          [[[[  496.,   466.,   436.],
             [  376.,   346.,   316.],
             [  256.,   226.,   196.]],
            [[ 1385.,  1403.,  1421.],
             [ 1457.,  1475.,  1493.],
             [ 1529.,  1547.,  1565.]]],
           [[[ -944.,  -974., -1004.],
             [-1064., -1094., -1124.],
             [-1184., -1214., -1244.]],
            [[ 2249.,  2267.,  2285.],
             [ 2321.,  2339.,  2357.],
             [ 2393.,  2411.,  2429.]]]], dtype=dtype)
        try:
            orig_params = layer.get_params()
            if kernels_first:
                layer.set_params([kernels, biases])
            else:
                layer.set_params([biases, kernels])
            layer_output = layer.fprop(inputs)
            assert layer_output.shape == true_output.shape, (
                'Layer fprop gives incorrect shaped output. '
                'Correct shape is {0} but returned shape is {1}.'
                .format(true_output.shape, layer_output.shape)
            )
            assert numpy.allclose(layer_output, true_output), (
                'Layer fprop does not give correct output. '
                'Correct output is {0}\n but returned output is {1}.'
                .format(true_output, layer_output)
            )
        finally:
            layer.set_params(orig_params)
        return True

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

        self.assertAlmostEqual(tst_accuracy, 0.326, delta=0.005)

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

        self.assertAlmostEqual(tst_accuracy, 0.136, delta=0.005)

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

        self.assertAlmostEqual(tst_accuracy, 0.19, delta=0.005)

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

        self.assertAlmostEqual(tst_accuracy, 0.059, delta=0.005)

    def test_model_fprop_fast(self):
        """ Ensure that back prop works when the conv layer has 1 below it """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=10, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=784))
        model.add_layer(ConvLinear(1, 3, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(ConvLinear(3, 3, image_shape=(24, 24), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Softmax(idim=1200, odim=10))

        for x, t in train_dp:
            start = time.clock()
            y = model.fprop(x)
            stop = time.clock()
            print("batch done in {}".format(stop - start))
        self.assertTrue(True)

    def test_model_bprop_fast(self):
        """ Ensure that back prop works when the conv layer has 1 below it """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=10, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=784))
        model.add_layer(ConvLinear(1, 3, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(ConvLinear(3, 3, image_shape=(24, 24), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(Softmax(idim=1200, odim=10))

        for x, t in train_dp:
            y = model.fprop(x)
            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)
            # do backward pass through the model
            start = time.clock()
            model.bprop(cost_grad)
            stop = time.clock()

            print("batch done in {}".format(stop - start))
        self.assertTrue(True)



