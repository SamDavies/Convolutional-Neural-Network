from unittest import TestCase

import numpy

from mlp.costs import CECost
from mlp.dataset import MNISTDataProvider
from mlp.layers import Sigmoid, Softmax, MLP
from mlp.optimisers import AutoEncoder, CrossEntropy


class AutoEncoderTestCase(TestCase):
    def test_pretrain(self):
        """ Ensure that  """
        # Given
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=1, randomize=False)

        train_dp.reset()
        valid_dp.reset()

        rng = numpy.random.RandomState([2015, 10, 10])
        rng_state = rng.get_state()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=100, rng=rng))
        model.add_layer(Sigmoid(idim=100, odim=100, rng=rng))
        model.add_layer(Softmax(idim=100, odim=10, rng=rng))
        auto_encoder = AutoEncoder(learning_rate=0.5, max_epochs=5)
        auto_encoder.pretrain(model, train_iter=train_dp)
        self.assertAlmostEqual(model.layers[0].W[0][0], 0.03, delta=0.005)
        self.assertAlmostEqual(model.layers[1].W[0][0], -0.03, delta=0.005)
        self.assertAlmostEqual(model.layers[2].W[0][0], 0.097, delta=0.005)


class CrossEntropyTestCase(TestCase):
    def test_pretrain(self):
        """ Ensure that """
        # Given
        train_dp = MNISTDataProvider(dset='train', batch_size=88, max_num_batches=1, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=88, max_num_batches=1, randomize=False)

        train_dp.reset()
        valid_dp.reset()

        rng = numpy.random.RandomState([2015, 10, 10])
        rng_state = rng.get_state()

        cost = CECost()
        model = MLP(cost=cost)
        model.add_layer(Sigmoid(idim=784, odim=100, rng=rng))
        model.add_layer(Sigmoid(idim=100, odim=100, rng=rng))
        model.add_layer(Sigmoid(idim=100, odim=100, rng=rng))
        model.add_layer(Softmax(idim=100, odim=10, rng=rng))
        auto_encoder = CrossEntropy(learning_rate=0.5, max_epochs=5)
        auto_encoder.pretrain(model, train_iter=train_dp)
        self.assertAlmostEqual(model.layers[0].W[0][0], 0.03, delta=0.005)
        self.assertAlmostEqual(model.layers[1].W[0][0], -0.011, delta=0.05)
        self.assertAlmostEqual(model.layers[2].W[0][0], 0.097, delta=0.005)
