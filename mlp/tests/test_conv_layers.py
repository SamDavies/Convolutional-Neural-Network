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

class LayerTypeTestCase(TestCase):
    def test_model_sigmoid(self):
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