cimport numpy
import numpy

DTYPE = numpy.float32

ctypedef numpy.float32_t DTYPE_t


def convolution_fprop_fast(
        numpy.ndarray[DTYPE_t, ndim=5] weights,
        numpy.ndarray[DTYPE_t, ndim=3] biases,
        unsigned int num_out_feat_maps,
        unsigned int kernel_shape_x,
        unsigned int kernel_shape_y,
        numpy.ndarray[DTYPE_t, ndim=4] inputs):
    """
    Implements a forward propagation for a convolution layer
    Note: filer means the same as kernel and convolution (correlation) of those with the input space
    produces feature maps (sometimes refereed to also as receptive fields). Also note, that
    feature maps are synonyms here to channels, and as such num_inp_channels == num_inp_feat_maps
    :param layer: the convolution layer
    :param inputs: 4D tensor of size (batch_size, num_out_feature_maps, num_rows_in_image, num_cols_in_image)
    :return: 4D tensor of size (batch_size, num_in_feature_maps, num_rows_in_layer, num_cols_in_layer)
    """
    assert weights.dtype == DTYPE and inputs.dtype == DTYPE

    cdef unsigned int num_batches = inputs.shape[0]
    cdef unsigned int num_rows_units = weights.shape[1]
    cdef unsigned int num_cols_units = weights.shape[2]
    cdef unsigned int num_input_feature_maps = inputs.shape[1]
    cdef unsigned int b, f, row_i, col_j, ifm, input_feature_map

    # make the activation tot be the size of the output
    cdef numpy.ndarray[DTYPE_t, ndim=4] activations = numpy.empty(
            (num_batches, num_out_feat_maps, num_rows_units, num_cols_units), dtype=numpy.float32)

    cdef numpy.ndarray[DTYPE_t, ndim=2] output = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)

    cdef numpy.ndarray[DTYPE_t, ndim=2] sub_img
    cdef numpy.ndarray[DTYPE_t, ndim=2] input_dot_weights

    for b in xrange(0, num_batches):
        for f in xrange(0, num_out_feat_maps):
            """ Given an image calculate the fprop for 1 feature maps """
            output = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)
            # go through each unit of this layer
            for row_i in range(0, num_rows_units):
                for col_j in range(0, num_cols_units):
                    for ifm in xrange(0, num_input_feature_maps):
                        # find the sum of the input * weight for every pixel in the kernel
                        sub_img = inputs[b][ifm][row_i:kernel_shape_x + row_i, col_j:kernel_shape_y + col_j]
                        input_dot_weights = numpy.multiply(sub_img, weights[f][row_i][col_j]) + biases[f][row_i][col_j]
                        # flatten and sum across all elements
                        output[row_i][col_j] += input_dot_weights.reshape(kernel_shape_x * kernel_shape_y).sum()
            # output shape is
            # - number of rows of units in this layer
            # - number of cols of units in this layer
            activations[b][f] = output
    return activations