from unittest import TestCase

from numpy.testing import assert_array_equal, assert_array_almost_equal

from mlp.conv import ConvLinear, ConvMaxPool2D
from mlp.costs import CECost
from mlp.dataset import MNISTDataProvider
from mlp.layers import MLP, Softmax, Sigmoid
from mlp.optimisers import SGDOptimiser
from mlp.schedulers import LearningRateFixed


class MaxPoolTestCase(TestCase):
    def test_model_fprop_max_pool(self):
        """ Ensure that 1 forward prop pass works """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=3, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(ConvLinear(1, 3, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1),irange=0.2))
        model.add_layer(ConvMaxPool2D(num_feat_maps=3, conv_shape=(24, 24), pool_shape=(2, 2)))
        model.add_layer(Sigmoid(idim=432, odim=4))

        x, t = train_dp.next()
        actual = model.fprop(x)
        assert_array_equal(model.layers[1].sudoW[6][0][7][0], [[0, 0], [0, 1]])
        assert_array_almost_equal(actual[0], [0.471462, 0.463854, 0.356474, 0.594162], verbose=True, decimal=2)

    def test_model_bprop_max_pool(self):
        """ Ensure that back prop works with pgrads """
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=3, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=3, randomize=True)
        test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=3, randomize=True)
        train_dp.reset()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(ConvLinear(1, 1, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(ConvMaxPool2D(num_feat_maps=1, conv_shape=(24, 24), pool_shape=(2, 2)))
        model.add_layer(Sigmoid(idim=144, odim=144))
        model.add_layer(Softmax(idim=144, odim=10))

        lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=2)
        optimiser = SGDOptimiser(lr_scheduler=lr_scheduler, dp_scheduler=None,
                                 l1_weight=0.0, l2_weight=0.0)

        optimiser.train(model, train_dp, valid_iterator=valid_dp)

        tst_cost, tst_accuracy = optimiser.validate(model, test_dp)

        self.assertAlmostEqual(tst_accuracy, 0.0766, delta=0.005)