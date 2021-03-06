cimport numpy
import numpy

DTYPE = numpy.float32

ctypedef numpy.float32_t DTYPE_t


def convolution_fprop_fast(
        numpy.ndarray[DTYPE_t, ndim=4] weights,
        numpy.ndarray[DTYPE_t, ndim=1] biases,
        unsigned int num_rows_units,
        unsigned int num_cols_units,
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
    cdef unsigned int num_input_feature_maps = inputs.shape[1]
    cdef unsigned int b, f, row_i, col_j, ifm, row_i_plus_kernel, col_j_plus_kernel

    # make the activation tot be the size of the output
    cdef numpy.ndarray[DTYPE_t, ndim=4] activations = numpy.zeros(
            (num_batches, num_out_feat_maps, num_rows_units, num_cols_units), dtype=numpy.float32)

    cdef numpy.ndarray[DTYPE_t, ndim=2] output = numpy.zeros((num_rows_units, num_cols_units), dtype=numpy.float32)

    activations = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(activations, 1, 0), 2, 1), 3, 2)
    inputs = numpy.rollaxis(inputs, 1, 0)

    for row_i in xrange(0, num_rows_units):
        for col_j in xrange(0, num_cols_units):
            row_i_plus_kernel = kernel_shape_x + row_i
            col_j_plus_kernel = kernel_shape_y + col_j
            for f in xrange(0, num_out_feat_maps):
                # go through each unit of this layer
                for ifm in xrange(0, num_input_feature_maps):
                    # sum along axis b
                    sum_along_b = numpy.sum(
                            (inputs[ifm][0:, row_i: row_i_plus_kernel, col_j:col_j_plus_kernel]
                             * weights[ifm][f]).reshape(num_batches, -1), axis=1)

                    activations[f][row_i][col_j] += sum_along_b
                activations[f][row_i][col_j] += biases[f]
    return numpy.rollaxis(activations, 3, 0)


def convolution_bprop_fast(
        numpy.ndarray[DTYPE_t, ndim=4] weights,
        numpy.ndarray[DTYPE_t, ndim=4] deltas,
        unsigned int image_shape_x,
        unsigned int image_shape_y,
        unsigned int num_inp_feat_maps):

    cdef unsigned int num_images = deltas.shape[0]

    cdef unsigned int num_out_feat_maps = weights.shape[1]
    cdef unsigned int num_rows_units = deltas.shape[2]
    cdef unsigned int num_cols_units = deltas.shape[3]
    cdef unsigned int kernel_shape_x = weights.shape[2]
    cdef unsigned int kernel_shape_y = weights.shape[3]

    deltas = deltas.T

    cdef unsigned int inp_feature_map_f, row_u, col_u, kernel_row_end, kernel_col_end, out_feature_map_f

    cdef numpy.ndarray[DTYPE_t, ndim=4] ograds = numpy.zeros(
            (num_images, num_inp_feat_maps, image_shape_x, image_shape_y), dtype=numpy.float32)

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
                    ograds[ifm][row_u:kernel_row_end, col_u:kernel_col_end, 0:] += (
                        weights[ifm][ofm][:, :, None] * deltas[col_u][row_u][ofm][0:]
                    )

    return numpy.rollaxis(ograds, 3, 0)


def convolution_pgrads_fast(
        numpy.ndarray[DTYPE_t, ndim=4] weights,
        numpy.ndarray[DTYPE_t, ndim=4] inputs,
        numpy.ndarray[DTYPE_t, ndim=4] deltas):
    cdef unsigned int num_input_feature_maps = inputs.shape[1]
    cdef unsigned int num_out_feat_maps = weights.shape[1]
    cdef unsigned int num_rows_units = deltas.shape[2]
    cdef unsigned int num_cols_units = deltas.shape[3]
    cdef unsigned int kernel_shape_x = weights.shape[2]
    cdef unsigned int kernel_shape_y = weights.shape[3]

    # set up deltas that be added to weight in the next step
    # i.e they have the same shape as weights
    cdef numpy.ndarray[DTYPE_t, ndim=4] grad_W = numpy.zeros(
            (weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]), dtype=numpy.float32)

    deltas = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(deltas, 1, 0), 2, 1), 3, 2)
    inputs = numpy.rollaxis(numpy.rollaxis(numpy.rollaxis(inputs, 1, 0), 2, 1), 3, 2)

    cdef unsigned int ifm, row_u, col_u, ofm, kernel_row_end, kernel_col_end

    for ifm in range(0, num_input_feature_maps):
        # for each row of units in this layer
        for row_u in range(0, num_rows_units):
            # for each col of units in this layer
            for col_u in range(0, num_cols_units):
                kernel_row_end = kernel_shape_x + row_u
                kernel_col_end = kernel_shape_y + col_u
                for ofm in range(0, num_out_feat_maps):
                    grad_W[ifm][ofm] += numpy.sum((
                        inputs[ifm][row_u:kernel_row_end, col_u:kernel_col_end][0:] * deltas[ofm][row_u][col_u][0:]
                    ), axis=2)

    return grad_W