import time
from unittest import TestCase

from mlp.conv import ConvLinear, ConvMaxPool2D
from mlp.costs import CECost
from mlp.dataset import MNISTDataProvider
from mlp.layers import MLP, Softmax, Sigmoid


class ConvSpeedTestCase(TestCase):
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
        model.add_layer(ConvLinear(1, 3, image_shape=(28, 28), kernel_shape=(5, 5), stride=(1, 1), irange=0.2))
        model.add_layer(ConvMaxPool2D(num_feat_maps=3, conv_shape=(24, 24), pool_shape=(2, 2)))
        model.add_layer(Sigmoid(idim=432, odim=432))
        model.add_layer(Softmax(idim=432, odim=10))

        for x, t in train_dp:
            full_start = time.clock()
            start = time.clock()
            y = model.fprop(x)
            stop = time.clock()
            print("batch fprop done in {}".format(stop - start))
            # compute the cost and grad of the cost w.r.t y
            cost = model.cost.cost(y, t)
            cost_grad = model.cost.grad(y, t)
            # do backward pass through the model
            start = time.clock()
            model.bprop(cost_grad)
            stop = time.clock()
            print("batch bprop done in {}".format(stop - start))

            print("batch done in {}".format(stop - full_start))
        self.assertTrue(True)