# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh


import numpy
import logging
from mlp.layers import Layer
import convx

logger = logging.getLogger(__name__)


def convolution_fprop_fast(weights, biases, num_rows_units, num_cols_units, num_out_feat_maps, kernel_shape_x, kernel_shape_y, inputs):
    """
    Implements a forward propagation for a convolution layer
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param layer: the convolution layer
    :param inputs: 4D tensor of size (batch_size, num_out_feature_maps, num_rows_in_image, num_cols_in_image)
    :return: 4D tensor of size (batch_size, num_in_feature_maps, num_rows_in_layer, num_cols_in_layer)
    """
    num_batches = inputs.shape[0]
    num_input_feature_maps = inputs.shape[1]

    # make the activation to be the size of the output
    activations = numpy.zeros((num_batches, num_out_feat_maps, num_rows_units, num_cols_units), dtype=numpy.float32)

    activations = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(activations, 1, 0), 2, 1), 3, 2)
    inputs = numpy.rollaxis(inputs, 1, 0)

    for row_i in xrange(0, num_rows_units):
        for col_j in xrange(0, num_cols_units):
            row_i_plus_kernel = kernel_shape_x + row_i
            col_j_plus_kernel = kernel_shape_y + col_j
            for f in xrange(0, num_out_feat_maps):
                # go through each unit of this layer
                for ifm in xrange(0, num_input_feature_maps):
                    sub_weights = weights[ifm][f]

                    sub_inputs = inputs[ifm][0:, row_i: row_i_plus_kernel, col_j:col_j_plus_kernel]
                    sub_inputs_w_b = (sub_inputs * sub_weights)
                    sub_inputs_w_b_flat = sub_inputs_w_b.reshape(num_batches, -1)

                    # sum along axis b
                    sum_along_b = numpy.sum(sub_inputs_w_b_flat, axis=1)

                    activations[f][row_i][col_j] += sum_along_b
                activations[f][row_i][col_j] += biases[f]
    return numpy.rollaxis(activations, 3, 0)


def convolution_bprop_fast(weights, deltas, image_shape_x, image_shape_y, num_inp_feat_maps):
    num_images = deltas.shape[0]

    num_out_feat_maps = weights.shape[1]
    num_rows_units = deltas.shape[2]
    num_cols_units = deltas.shape[3]
    kernel_shape_x = weights.shape[2]
    kernel_shape_y = weights.shape[3]

    deltas = deltas.T

    ograds = numpy.zeros((num_images, num_inp_feat_maps, image_shape_x, image_shape_y), dtype=numpy.float32)

    ograds = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(ograds, 1, 0), 2, 1), 3, 2)

    # for each input feature map
    for ifm in range(0, num_inp_feat_maps):
        # for each row of units in this layer
        for row_u in range(0, num_rows_units):
            # for each col of units in this layer
            for col_u in range(0, num_cols_units):
                kernel_row_end = kernel_shape_x + row_u
                kernel_col_end = kernel_shape_y + col_u
                # for each feature map in this layer
                for ofm in range(0, num_out_feat_maps):
                    # for each image
                    unit_delta = deltas[col_u][row_u][ofm][0:]
                    weights_delta = weights[ifm][ofm][:, :, None] * unit_delta
                    # find the portion in the image which is effected by this unit
                    image_segment = ograds[ifm][row_u:kernel_row_end,
                                    col_u:kernel_col_end, 0:]
                    image_segment += weights_delta

    return numpy.rollaxis(ograds, 3, 0)


def convolution_pgrads_fast(weights, inputs, deltas):
    num_input_feature_maps = inputs.shape[1]

    num_out_feat_maps = weights.shape[1]
    num_rows_units = deltas.shape[2]
    num_cols_units = deltas.shape[3]
    kernel_shape_x = weights.shape[2]
    kernel_shape_y = weights.shape[3]

    # set up deltas that be added to weight in the next step
    # i.e they have the same shape as weights
    grad_W = numpy.zeros(weights.shape, dtype=numpy.float32)

    deltas = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(deltas, 1, 0), 2, 1), 3, 2)
    inputs = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(inputs, 1, 0), 2, 1), 3, 2)

    for ifm in range(0, num_input_feature_maps):
        # for each row of units in this layer
        for row_u in range(0, num_rows_units):
            # for each col of units in this layer
            for col_u in range(0, num_cols_units):
                kernel_row_end = kernel_shape_x + row_u
                kernel_col_end = kernel_shape_y + col_u
                for ofm in range(0, num_out_feat_maps):
                    unit_delta = deltas[ofm][row_u][col_u][0:]
                    input_kernel = inputs[ifm][row_u:kernel_row_end, col_u:kernel_col_end][0:]
                    grad = input_kernel * unit_delta
                    sum_grad = numpy.sum(grad, axis=2)
                    grad_W[ifm][ofm] += sum_grad
    return grad_W


class ConvLinear(Layer):
    def __init__(self,
                 num_inp_feat_maps,
                 num_out_feat_maps,
                 image_shape=(28, 28),
                 kernel_shape=(5, 5),
                 stride=(1, 1),
                 irange=0.2,
                 rng=None,
                 conv_fwd=convolution_fprop_fast,
                 conv_bck=convolution_fprop_fast,
                 conv_grad=convolution_fprop_fast):
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


        self.num_rows_units = (image_shape[0] - kernel_shape[0] + 1)
        self.num_cols_units = (image_shape[1] - kernel_shape[1] + 1)

        # number of output feature maps
        # unit row
        # unit col
        # kernel row
        # kernel col
        self.W = numpy.array(
                self.rng.uniform(
                        -irange, irange,
                        (num_inp_feat_maps,
                         num_out_feat_maps,
                         kernel_shape[0],
                         kernel_shape[1])
                ), dtype=numpy.float32)

        self.b = numpy.zeros((
            num_out_feat_maps,
        ), dtype=numpy.float32)

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

        inputs = numpy.array(inputs, dtype=numpy.float32)
        self.W = numpy.array(self.W, dtype=numpy.float32)
        self.b = numpy.array(self.b, dtype=numpy.float32)
        activations = convx.convolution_fprop_fast(
                self.W, self.b, self.num_rows_units, self.num_cols_units, self.num_out_feat_maps,
                self.kernel_shape[0], self.kernel_shape[1], inputs
        )

        # output shape is
        # - batch size
        # - number of output feature maps
        # - number of rows of units in this layer
        # - number of cols of units in this layer
        # and the non-convolutional layers will flatter this to
        # - number of batches
        # - the rest
        return activations

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

        # shape the igrads into this layer size
        deltas_square = igrads.reshape(igrads.shape[0], self.num_out_feat_maps, self.num_rows_units, self.num_cols_units)

        self.W = numpy.array(self.W, dtype=numpy.float32)
        deltas_square = numpy.array(deltas_square, dtype=numpy.float32)
        ograds = convx.convolution_bprop_fast(
                self.W, deltas_square, self.image_shape[0], self.image_shape[1], self.num_inp_feat_maps
        )

        # flatten the image in ograds
        ograds_flat = ograds.reshape(igrads.shape[0], -1)

        # shape of ograds:
        # - batch size
        # - num input feature map
        # - input image rows
        # - input image cols
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
        # shape the input into - batch_size X output_feature_maps X image_x X image_y
        inputs = inputs.reshape((inputs.shape[0], self.num_inp_feat_maps, self.image_shape[0], self.image_shape[1]))

        # shape the deltas into - batch_size X input_feature_maps X unit_rows X unit_cols
        deltas = deltas.reshape((deltas.shape[0], self.num_out_feat_maps, self.num_rows_units, self.num_cols_units))

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

        self.W = numpy.array(self.W, dtype=numpy.float32)
        inputs = numpy.array(inputs, dtype=numpy.float32)
        deltas = numpy.array(deltas, dtype=numpy.float32)
        grad_W = convx.convolution_pgrads_fast(self.W, inputs, deltas)

        b_deltas = deltas.reshape((deltas.shape[0], self.num_out_feat_maps, self.num_rows_units * self.num_cols_units))
        b_deltas_sum_batch = numpy.sum(b_deltas, axis=0)
        # make the gradients for the bias square
        grad_b = numpy.sum(b_deltas_sum_batch, axis=1) + l2_b_penalty + l1_b_penalty

        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        self.W = params[0]
        self.b = params[1]

    def get_name(self):
        return 'convlinear'


class ConvSigmoid(ConvLinear):
    def fprop(self, inputs):
        # get the linear activations
        a = super(ConvSigmoid, self).fprop(inputs)
        # stabilise the exp() computation in case some values in
        # 'a' get very negative.
        numpy.clip(a, -30.0, 30.0, out=a)
        h = 1.0 / (1 + numpy.exp(-a))
        return h

    def bprop(self, h, igrads):
        h = h.reshape(igrads.shape)
        dsigm = h * (1.0 - h)
        deltas = igrads * dsigm
        ___, ograds = super(ConvSigmoid, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'bce':
            return super(ConvSigmoid, self).bprop(h=h, igrads=igrads)
        else:
            raise NotImplementedError('Sigmoid.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'convsigmoid'


class ConvRelu(ConvLinear):
    def fprop(self, inputs):
        # get the linear activations
        a = super(ConvRelu, self).fprop(inputs)
        h = numpy.clip(a, 0, 20.0)
        # h = numpy.maximum(a, 0)
        return h

    def bprop(self, h, igrads):
        h = h.reshape(igrads.shape)
        deltas = (h > 0) * igrads
        ___, ograds = super(ConvRelu, self).bprop(h=None, igrads=deltas)
        return deltas, ograds

    def bprop_cost(self, h, igrads, cost):
        raise NotImplementedError('Relu.bprop_cost method not implemented '
                                  'for the %s cost' % cost.get_name())

    def get_name(self):
        return 'convrelu'


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
        self.num_feat_maps = num_feat_maps
        self.conv_shape = conv_shape
        self.pool_shape = pool_shape
        self.pool_stride = pool_stride

        self.num_unit_rows = conv_shape[0]/pool_shape[0]
        self.num_unit_cols = conv_shape[1]/pool_shape[1]

        super(ConvMaxPool2D, self).__init__(rng=None)
        self.sudoW = None

    def fprop(self, inputs):
        # Note that the sudoW is not used to calculate the activations
        # It is set to 1 for the max input in the pool
        self.sudoW = numpy.zeros((
            inputs.shape[0],
            self.num_feat_maps,
            self.num_unit_rows,
            self.num_unit_cols,
            self.pool_shape[0],
            self.pool_shape[1]
        ), dtype=numpy.float32)

        num_batches = inputs.shape[0]
        num_feat_maps = inputs.shape[1]
        num_rows_units = self.sudoW.shape[2]
        num_cols_units = self.sudoW.shape[3]
        pool_shape_x = self.sudoW.shape[4]
        pool_shape_y = self.sudoW.shape[5]
        pool_flat_shape = pool_shape_x * pool_shape_y

        # make the activation to be the size of the output
        activations = numpy.zeros((num_batches, self.num_feat_maps, self.num_unit_rows, self.num_unit_rows), dtype=numpy.float32)
        inputs = numpy.rollaxis(inputs, 1, 0)
        activations = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(activations, 1, 0), 2, 1), 3, 2)
        self.sudoW = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(self.sudoW, 1, 0), 2, 1), 3, 2), 4, 3), 5, 4)

        for row_i in xrange(0, num_rows_units):
            for col_j in xrange(0, num_cols_units):
                pool_row_i = row_i * pool_shape_x
                pool_col_j = col_j * pool_shape_y
                row_i_plus_pool = pool_row_i + pool_shape_x
                col_j_plus_pool = pool_col_j + pool_shape_y
                for f in xrange(0, num_feat_maps):
                    sub_inputs = inputs[f][0:, pool_row_i: row_i_plus_pool, pool_col_j:col_j_plus_pool]
                    flat_sub_inputs = numpy.reshape(sub_inputs, (num_batches, pool_flat_shape))
                    sub_maxes = numpy.max(flat_sub_inputs, axis=1)
                    activations[f][row_i][col_j][0:] = sub_maxes

                    # record the max for this image
                    index_max = flat_sub_inputs.argmax(axis=1)
                    weight_col_row = self.sudoW[f][row_i][col_j]
                    flat_pool = numpy.reshape(weight_col_row, (self.sudoW.shape[3]*self.sudoW.shape[4], self.sudoW.shape[5]))
                    for image_i in xrange(0, num_batches):
                        flat_pool[index_max[image_i]][image_i] = 1

        self.sudoW = numpy.rollaxis(self.sudoW, 5, 0)
        return numpy.rollaxis(activations, 3, 0)

    def bprop(self, h, igrads):
        # shape the igrads into this layer size
        deltas = igrads.reshape(igrads.shape[0], self.num_feat_maps, self.num_unit_rows, self.num_unit_cols)

        num_batches = self.sudoW.shape[0]
        num_feat_maps = self.sudoW.shape[1]
        num_rows_units = self.sudoW.shape[2]
        num_cols_units = self.sudoW.shape[3]
        pool_shape_x = self.sudoW.shape[4]
        pool_shape_y = self.sudoW.shape[5]
        pool_flat_shape = pool_shape_x * pool_shape_y

        deltas = deltas.T

        ograds = numpy.zeros((num_batches, num_feat_maps, num_rows_units * pool_shape_x, num_cols_units * pool_shape_y), dtype=numpy.float32)

        # deltas = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(deltas_square, 1, 0), 2, 1), 3, 2)
        # ograds = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(ograds, 1, 0), 2, 1), 3, 2)

        for row_i in xrange(0, num_rows_units):
            for col_j in xrange(0, num_cols_units):
                pool_row_i = row_i * pool_shape_x
                pool_col_j = col_j * pool_shape_y
                row_i_plus_pool = pool_row_i + pool_shape_x
                col_j_plus_pool = pool_col_j + pool_shape_y
                for f in xrange(0, num_feat_maps):
                    for image_i in xrange(0, num_batches):
                        delta = deltas[col_j][row_i][f][image_i]
                        w = self.sudoW[image_i][f][row_i][col_j].T
                        ograds[image_i][f][pool_row_i: row_i_plus_pool, pool_col_j: col_j_plus_pool] = w * delta

        # ograds = numpy.rollaxis(ograds, 3, 0)
        # flatten the image in ograds
        ograds_flat = ograds.reshape(igrads.shape[0], -1)

        return igrads, ograds_flat

    def get_params(self):
        return []

    def pgrads(self, inputs, deltas, **kwargs):
        return []

    def set_params(self, params):
        pass

    def get_name(self):
        return 'convmaxpool2d'
