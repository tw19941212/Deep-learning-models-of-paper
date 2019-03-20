# @Author : tw19941212
# @Datetime : 2019/01/17 14:50
# @Last Modify Time : 2019/01/17 15:19

from keras.engine import Layer, InputSpec
from keras.layers import Flatten, Lambda
import tensorflow as tf


class KMaxPooling(Layer):
    """
    extracts the k-highest activations from a sequence

    # Input shape
        3D tensor with shape `(batch_size, step_size, input_features)`
    # Output shape
        2D tensor with shape `(batch_size, input_features*k)`
    # Example
        x = LSTM(64, return_sequences=True)(input_words)
        x = KMaxPooling(3)(x)
    """

    def __init__(self, k=1, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def call(self, x):
        # swap last two dimensions since top_k will be applied along the last dimensions
        shifted_x = tf.transpose(x, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_x, self.k, sorted=True)[0]

        return Flatten()(top_k)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]*self.k

# other implementation
# Lambda(lambda x: Flatten()(tf.nn.top_k(tf.transpose(x,[0,2,1]),k=2)[0]))
