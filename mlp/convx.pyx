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
    cdef unsigned int b, f, row_i, col_j, ifm, input_feature_map, row_i_plus_kernel, col_j_plus_kernel

    # make the activation tot be the size of the output
    cdef numpy.ndarray[DTYPE_t, ndim=4] activations = numpy.zeros(
            (num_batches, num_out_feat_maps, num_rows_units, num_cols_units), dtype=numpy.float32)

    cdef numpy.ndarray[DTYPE_t, ndim=2] output = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)

    cdef numpy.ndarray[DTYPE_t, ndim=2] sub_img
    cdef numpy.ndarray[DTYPE_t, ndim=2] input_dot_weights

    activations = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(activations, 1, 0), 2, 1), 3, 2)
    inputs = numpy.rollaxis(inputs, 1, 0)

    for row_i in xrange(0, num_rows_units):
        for col_j in xrange(0, num_cols_units):
            row_i_plus_kernel = kernel_shape_x + row_i
            col_j_plus_kernel = kernel_shape_y + col_j
            for f in xrange(0, num_out_feat_maps):
                # go through each unit of this layer
                for ifm in xrange(0, num_input_feature_maps):
                    activations[f][row_i][col_j] += numpy.sum(
                            ((inputs[ifm][0:, row_i: row_i_plus_kernel, col_j:col_j_plus_kernel] * weights[f][row_i][col_j])
                             + biases[f][row_i][col_j]
                    ).reshape(num_batches, -1), axis=1)
    return numpy.rollaxis(activations, 3, 0)