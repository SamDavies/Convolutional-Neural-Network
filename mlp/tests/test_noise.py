from unittest import TestCase

import numpy

from mlp.dataset import MNISTDataProvider, add_batches
from mlp.noise import DropoutNoise, NoiseMaker


class NoiseTestCase(TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState([2015, 10, 10])
        self.rng_state = self.rng.get_state()

        self.train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=1, randomize=True)
        self.train_dp.reset()

        noise_type = DropoutNoise(dropout_prob=0.5)
        noise_maker = NoiseMaker(data_set=self.train_dp, num_batches=1, noise=noise_type)
        new_batches = noise_maker.make_examples(rng=self.rng)
        add_batches(self.train_dp, new_batches)

    def test_add_noise(self):

        self.assertEqual(len(self.train_dp.x), 50100)
        self.assertEqual(self.train_dp.t[99], self.train_dp.t[50099])

    def test_can_get_new_examples(self):

        self.assertEqual(len(self.train_dp.x), 50100)
        self.assertEqual(self.train_dp.t[99], self.train_dp.t[50099])
