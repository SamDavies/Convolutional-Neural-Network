# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh


import numpy
import logging
from mlp.layers import Layer

logger = logging.getLogger(__name__)

"""
You have been given some very initial skeleton below. Feel free to build on top of it and/or
modify it according to your needs. Just notice, you can factor out the convolution code out of
the layer code, and just pass (possibly) different conv implementations for each of the stages
in the model where you are expected to apply the convolutional operator. This will allow you to
keep the layer implementation independent of conv operator implementation, and you can easily
swap it layer, for example, for more efficient implementation if you came up with one, etc.
"""


def my1_conv2d(image, kernels, strides=(1, 1)):
    """
    Implements a 2d valid convolution of kernels with the image
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param image: 4D tensor of sizes (batch_size, num_input_channels, img_shape_x, img_shape_y)
    :param filters: 4D tensor of filters of size (num_inp_feat_maps, num_out_feat_maps, kernel_shape_x, kernel_shape_y)
    :param strides: a tuple (stride_x, stride_y), specifying the shift of the kernels in x and y dimensions
    :return: 4D tensor of size (batch_size, num_out_feature_maps, feature_map_shape_x, feature_map_shape_y)
    """
    raise NotImplementedError('Write me!')


class ConvLinear(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=my1_conv2d,
                 conv_bck=my1_conv2d,
                 conv_grad=my1_conv2d):
        """

        :param num_inp_feat_maps: int, a number of input feature maps (channels)
        :param num_out_feat_maps: int, a number of output feature maps (channels)
        :param image_shape: tuple, a shape of the image
        :param kernel_shape: tuple, a shape of the kernel
        :param stride: tuple, shift of kernels in both dimensions
        :param irange: float, initial range of the parameters
        :param rng: RandomState object, random number generator
        :param conv_fwd: handle to a convolution function used in fwd-prop
        :param conv_bck: handle to a convolution function used in backward-prop
        :param conv_grad: handle to a convolution function used in pgrads
        :return:
        """

        super(ConvLinear, self).__init__(rng=rng)

        self.num_inp_feat_maps = num_inp_feat_maps
        self.num_out_feat_maps = num_out_feat_maps

        self.image_shape = image_shape
        self.kernel_shape = kernel_shape
        self.stride = stride

        self.conv_fwd = conv_fwd
        self.conv_bck = conv_bck
        self.conv_grad = conv_grad

        # output dimensions is the number of kernels which fit into the image
        # self.idim = kernel_shape[0] * kernel_shape[1]
        # self.odim = (image_shape[0] - kernel_shape[0] + 1) * (image_shape[1] - kernel_shape[1] + 1)
        # make an array of kernels for each feature

        # number of output feature maps
        # unit row
        # unit col
        # kernel row
        # kernel col
        self.W = self.rng.uniform(
                -irange, irange,
                (num_out_feat_maps,
                 (image_shape[0] - kernel_shape[0] + 1),
                 (image_shape[1] - kernel_shape[1] + 1),
                 kernel_shape[0],
                 kernel_shape[1])
        )

        self.b = numpy.zeros((
            num_out_feat_maps,
            (image_shape[0] - kernel_shape[0] + 1),
            (image_shape[1] - kernel_shape[1] + 1)
        ), dtype=numpy.float32)

    # def get_weights(self):
    #     """
    #     Reshape the weight so we can apply nice transformations to it
    #     """
    #     feature_map_x = (self.image_shape[0] - self.kernel_shape[0] + 1)
    #     feature_map_y = (self.image_shape[1] - self.kernel_shape[1] + 1)
    #     return numpy.swapaxes(self.W, 0, 1).reshape((feature_map_x, feature_map_y, 5, 5))
    #
    # def get_bias(self):
    #     """
    #     Reshape the bias so we can apply nice transformations to it
    #     """
    #     feature_map_x = (self.image_shape[0] - self.kernel_shape[0] + 1)
    #     feature_map_y = (self.image_shape[1] - self.kernel_shape[1] + 1)
    #     return self.b.reshape((feature_map_x, feature_map_y))

    def fprop(self, inputs):
        """
        The input will have shape
         - number of batches
         - number of input feature maps
         - number of pixels
        :param inputs: the batch to forward propagate
        :return: the activations for the whole batch
        """
        # reshape if coming from a non-convulsion layer
        if inputs.ndim == 2:
            inputs = numpy.expand_dims(inputs, axis=1)

        # reshape the pixels to be 2D making 4D inputs
        inputs = inputs.reshape((inputs.shape[0], inputs.shape[1], self.image_shape[0], self.image_shape[1]))

        num_batches = inputs.shape[0]
        num_rows_units = self.W.shape[1]
        num_cols_units = self.W.shape[2]
        # make the activation tot be the size of the output
        activations = numpy.empty((num_batches, self.num_out_feat_maps, num_rows_units, num_cols_units), dtype=numpy.float32)

        for b in xrange(0, num_batches):
            for f in xrange(0, self.num_out_feat_maps):
                activations[b][f] = self.fprop_single_feature_map(inputs[b], f)

        # output shape is
        # - number of batches
        # - number of output feature maps
        # - number of rows of units in this layer
        # - number of cols of units in this layer
        # and the non-convolutional layers will flatter this to
        # - number of batches
        # - the rest
        return activations

    def fprop_single_feature_map(self, feature_maps, f):
        """
        Given an image calculate the fprop for 1 feature maps
        :param feature_maps: the 2D image data X feature maps making 3D
        :return: the predicted output of this layer
        """
        feature_map = feature_maps[f]
        # the pixels of the input image
        num_rows_units = self.W.shape[1]
        num_cols_units = self.W.shape[2]
        output = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)
        # go through each unit of this layer
        for row_i in range(0, num_rows_units):
            for col_j in range(0, num_cols_units):
                # find the sum of the input * weight for every pixel in the kernel
                sub_img = feature_map[row_i:self.kernel_shape[0] + row_i, col_j:self.kernel_shape[1] + col_j]
                input_dot_weights = numpy.multiply(sub_img, self.W[f][row_i][col_j]) + self.b[f][row_i][col_j]
                # flatten and sum across all elements
                output[row_i][col_j] = input_dot_weights.reshape(self.kernel_shape[0] * self.kernel_shape[1]).sum()

        # here f() is an identity function, so just return a linear transformation
        # output shape is
        # - number of rows of units in this layer
        # - number of cols of units in this layer
        return output

    def bprop(self, h, igrads):
        """
        h shape is:
            - batch size
            - units in this layer
        :param h: the activations produced in the forward pass of this layer
        :param igrads: igrads, error signal (or gradient) flowing to the layer, note,
               this in general case does not corresponds to 'deltas' used to update
               the layer's parameters, to get deltas ones need to multiply it with
               the dh^i/da^i derivative
        :return:
        """
        deltas = igrads

        # shape the igrades into this layer size
        deltas_square = igrads.reshape(igrads.shape[0], self.W.shape[1], self.W.shape[2])

        ograds = numpy.zeros((igrads.shape[0], self.image_shape[0], self.image_shape[1]), dtype=numpy.float32)

        # for each image
        for image_i in range(0, igrads.shape[0]):
            # for each row of units in this layer
            for row_u in range(0, self.W.shape[1]):
                # for each col of units in this layer
                for col_u in range(0, self.W.shape[2]):
                        unit_delta = deltas_square[image_i][row_u][col_u]
                        # find the portion in the image which is effected by this unit
                        image_segment = ograds[image_i][row_u:self.kernel_shape[0] + row_u, col_u:self.kernel_shape[1] + col_u]
                        image_segment += unit_delta

        # flatten the image in ograds
        ograds_flat = ograds.reshape(igrads.shape[0], -1)

        # shape of ograds:
        # - batch size
        # - units in this layer
        # shape of deltas same as igrads
        return deltas, ograds_flat

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('ConvLinear.bprop_cost method not implemented')

    def pgrads(self, inputs, deltas, l1_weight=0, l2_weight=0):
        """
        Using the deltas, calculate which weights are effected by each unit.
        Make a sudo weights from the deltas by summing the deltas for each corresponding weight.
        Also sum acroos layer above's inputs
        :param inputs: batch size X
        :param deltas:
        :param l1_weight:
        :param l2_weight:
        :return:
        """
        # shape the input into an image
        inputs = inputs.reshape((inputs.shape[0], self.image_shape[0], self.image_shape[1]))

        # shape the deltas into row by cols for units of this layer
        deltas = deltas.reshape((deltas.shape[0], self.W.shape[1], self.W.shape[2]))

        # you could basically use different scalers for biases
        # and weights, but it is not implemented here like this
        l2_W_penalty, l2_b_penalty = 0, 0
        if l2_weight > 0:
            l2_W_penalty = l2_weight * self.W
            l2_b_penalty = l2_weight * self.b

        l1_W_penalty, l1_b_penalty = 0, 0
        if l1_weight > 0:
            l1_W_penalty = l1_weight * numpy.sign(self.W)
            l1_b_penalty = l1_weight * numpy.sign(self.b)

        # set up deltas that be added to weight in the next step
        # i.e they have the same shape as weights
        grad_W = numpy.zeros(self.W.shape, dtype=numpy.float32)

        # for each row of units in this layer
        for row_u in range(0, self.W.shape[1]):
            # for each col of units in this layer
            for col_u in range(0, self.W.shape[2]):
                # for each image
                for image_i in range(0, inputs.shape[0]):
                    unit_delta = deltas[image_i][row_u][col_u]
                    input_kernel = inputs[image_i][row_u:self.kernel_shape[0] + row_u, col_u:self.kernel_shape[1] + col_u]
                    grad_W[0][row_u][col_u] += input_kernel * unit_delta

        grad_b_flat = numpy.sum(deltas, axis=0) + l2_b_penalty + l1_b_penalty
        # make the gradients for the bias square
        grad_b = grad_b_flat.reshape((self.b.shape[1], self.b.shape[1]))

        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convlinear'


# you can derive here particular non-linear implementations:
# class ConvSigmoid(ConvLinear):
# ...


class ConvMaxPool2D(Layer):
    def __init__(self,
                 num_feat_maps,
                 conv_shape,
                 pool_shape=(2, 2),
                 pool_stride=(2, 2)):
        """

        :param conv_shape: tuple, a shape of the lower convolutional feature maps output
        :param pool_shape: tuple, a shape of pooling operator
        :param pool_stride: tuple, a strides for pooling operator
        :return:
        """

        super(ConvMaxPool2D, self).__init__(rng=None)
        raise NotImplementedError()

    def fprop(self, inputs):
        raise NotImplementedError()

    def bprop(self, h, igrads):
        raise NotImplementedError()

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'
