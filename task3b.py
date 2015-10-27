import numpy

#Seed a random number generator running the below cell, but do **not** modify the seed.
rng = numpy.random.RandomState([2015,10,10])
rng_state = rng.get_state()

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in
        IPython Notebook. """

    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")

            for col in row:
                html.append("<td>{0}</td>".format(col))

            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

import logging
import matplotlib.pyplot as plt

from mlp.costs import MSECost, CECost  # import the cost we want to use for optimisation
from mlp.dataset import MNISTDataProvider  # import data provider
from mlp.layers import MLP, Sigmoid, Softmax  # import required layer types
from mlp.optimisers import SGDOptimiser  # import the optimiser
from mlp.schedulers import LearningRateFixed

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

learning_rates = [0.5, 0.2, 0.1, 0.05, 0.01, 0.005]

test_errors = []

fig, axis_array = plt.subplots(1, 2, figsize=(15, 4))

for learning_rate in learning_rates:
    # define the model structure
    cost = CECost()
    model = MLP(cost=cost)
    model.add_layer(Sigmoid(idim=784, odim=100, rng=rng))
    model.add_layer(Softmax(idim=100, odim=10, rng=rng))
    # one can stack more layers here
    # define the optimiser, here stochasitc gradient descent
    # with fixed learning rate and max_epochs as stopping criterion
    lr_scheduler = LearningRateFixed(learning_rate=learning_rate, max_epochs=30)
    optimiser = SGDOptimiser(lr_scheduler=lr_scheduler)

    logger.warning('Initialising data providers...')
    train_dp = MNISTDataProvider(dset='train', batch_size=100, max_num_batches=-10, randomize=True)
    valid_dp = MNISTDataProvider(dset='valid', batch_size=100, max_num_batches=-10, randomize=False)

    logger.warning('Training started...')
    train_stats, valid_stats = optimiser.train(model, train_dp, valid_dp)

    logger.warning('Testing the model on test set...')
    test_dp = MNISTDataProvider(dset='eval', batch_size=100, max_num_batches=-10, randomize=False)
    cost, accuracy = optimiser.validate(model, test_dp)
    logger.warning('MNIST test set accuracy is %.2f %% (cost is %.3f)' % (accuracy * 100., cost))

    train_errors = [train_stat[0] for train_stat in train_stats]
    axis_array[0].plot(train_errors, label='Eta %s' % (str(learning_rate)))

    valid_errors = [valid_stat[0] for valid_stat in valid_stats]
    axis_array[1].plot(valid_errors, label='Eta %s' % (str(learning_rate)))

    test_errors.append(1-accuracy)

# show to line graphs using the training and validation results
axis_array[0].set_xlabel('epoch')
axis_array[0].set_ylabel('training error')
# fig.legend(lines, labels, loc = (0.5, 0), ncol=5 )
axis_array[0].grid()

axis_array[1].set_xlabel('epoch')
axis_array[1].set_ylabel('validation error')
# axis_array[1].legend()
axis_array[1].grid()

plt.show()

# print a html table of the test errors
table = ListTable()
table.append(["Epoch " + str(learning_rate) for learning_rate in learning_rates])
table.append(test_errors)
table