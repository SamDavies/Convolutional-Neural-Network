

def get_dropout_noise_examples(num_batches, batch_size, dropout_prob, data_set, rng):
    # create an array which is the size of the number of batches batches
    new_example_batches = [None] * num_batches
    # go through all of the baches
    for ith_batch in xrange(num_batches):
        # get the images and the digits for this batch
        x, t = data_set.next()
        # create an array which is the size of this batch
        new_examples = [None] * batch_size
        # go through each image in the batch
        for i in xrange(batch_size):
            img = x[i]
            digit = t[i]
            # add the noise
            d = rng.binomial(1, dropout_prob, img.shape)
            # save the image
            new_examples[i] = (d*img)
        # add the batch to the new batches
        new_example_batches[ith_batch] = new_examples
    return new_example_batches
