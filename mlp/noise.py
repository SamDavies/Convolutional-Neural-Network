import numpy


class NoiseMaker(object):
    def __init__(self, data_set, num_batches, noise):
        self.data_set = data_set
        self.num_batches = num_batches
        self.noise = noise

    def make_examples(self, rng):
        # create an array which is the size of the number of batches
        new_example_batches = [None] * self.num_batches
        # go through all of the batches
        for ith_batch in xrange(self.num_batches):
            # get the images and the digits for this batch
            x, t = self.data_set.next()
            batch_size = len(x)
            # create an array which is the size of this batch
            new_examples = [None] * batch_size
            # get the original digits
            new_digits = self.data_set.t[ith_batch*batch_size: (ith_batch+1)*batch_size]
            # go through each image in the batch
            for i in xrange(batch_size):
                img = x[i]
                digit = t[i]
                # add the noise and save the image
                new_examples[i] = self.noise.apply_noise(img, rng)
            # add the batch to the new batches
            new_example_batches[ith_batch] = (new_examples, new_digits)
        return new_example_batches


class AbstractNoise(object):
    def apply_noise(self, img, rng):
        raise NotImplementedError("You can't use abstract noise.")


class DropoutNoise(AbstractNoise):
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob

    def apply_noise(self, img, rng):
        d = rng.binomial(1, self.dropout_prob, img.shape)
        return d*img


class RotationNoise(AbstractNoise):
    def __init__(self, dropout_prob):
        self.dropout_prob = dropout_prob

    def apply_noise(self, img, rng):
        img_shape = numpy.sqrt(len(img))
        img = img.reshape((img_shape, img_shape))
        img = numpy.rot90(img)
        img = img.reshape((img_shape * img_shape))
        return img
