import numpy

# Seed a random number generator running the below cell, but do **not** modify the seed.
rng = numpy.random.RandomState([2015, 10, 10])
rng_state = rng.get_state()

import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mlp.costs import CECost
from mlp.dataset import MNISTDataProvider
from mlp.layers import MLP, Sigmoid, Softmax
from mlp.optimisers import SGDOptimiser
from mlp.schedulers import LearningRateFixed

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

cost = CECost()
model = MLP(cost=cost)
model.add_layer(Sigmoid(idim=784, odim=100, rng=rng))
model.add_layer(Softmax(idim=100, odim=10, rng=rng))

lr_scheduler = LearningRateFixed(learning_rate=0.5, max_epochs=2)
optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

logger.warning('Initialising data providers...')
train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True)
valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=-10, randomize=False)

train_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)

test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
cost, accuracy = optimiser.validate(model, test_dp)
logger.warning('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))

# extract the input weights for each hidden unit and show as an image
sigmoid_weights = model.layers[0].W
fig, axis_array = plt.subplots(10, 10, figsize=(15, 15))

for i in xrange(0, 10):
    for j in xrange(0, 10):
        k = i * 10 + j
        inputs_of_hidden_unit = numpy.array([input_unit[k] for input_unit in sigmoid_weights])
        axis_array[i, j].imshow(inputs_of_hidden_unit.reshape(28, 28), cmap=cm.Greys_r)
plt.show()


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** numpy.ceil(numpy.log(numpy.abs(matrix).max()) / numpy.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in numpy.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = numpy.sqrt(numpy.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


softmax_weights = model.layers[1].W
inputs_of_top_unit = numpy.array([input_unit[0] for input_unit in softmax_weights])
hinton(inputs_of_top_unit.reshape(10, 10))
plt.show()
