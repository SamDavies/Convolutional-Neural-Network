from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mlp.dataset import MNISTDataProvider, add_batches
from mlp.layers import Sigmoid, Softmax, Linear
from mlp.noise import DropoutNoise, NoiseMaker
from mlp.optimisers import AutoEncoder

__author__ = 'Sam Davies'


class AutoEncoderTestCase(TestCase):
    def test_train_layer(self):
        """ Ensure that a single layer can be trained """
        # Given
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1, randomize=True)
        valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=1, randomize=False)

        train_dp.reset()
        valid_dp.reset()

        rng = numpy.random.RandomState([2015, 10, 10])
        rng_state = rng.get_state()
        layer1 = Linear(idim=784, odim=100, irange=0.2, rng=rng)
        layer2 = Linear(idim=100, odim=784, irange=0.2, rng=rng)
        model = AutoEncoder.train_layer(layers=[layer1, layer2], train_iter=train_dp, valid_iter=valid_dp)
        self.assertTrue(False)


class NoiseTestCase(TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState([2015, 10, 10])
        self.rng_state = self.rng.get_state()

    def test_add_noise(self):
        train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1, randomize=True)
        train_dp.reset()

        noise_type = DropoutNoise(dropout_prob=0.5)
        noise_maker = NoiseMaker(data_set=train_dp, num_batches=1, noise=noise_type)
        new_batches = noise_maker.make_examples(rng=self.rng)
        add_batches(train_dp, new_batches)

        self.assertEqual(len(train_dp.x), 50100)
        self.assertEqual(train_dp.t[99], train_dp.t[50099])
