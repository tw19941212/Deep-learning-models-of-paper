# @Author : tw19941212
# @Datetime : 2019/01/16 21:42
# @Last Modify Time : 2019/01/17 11:37
# @paper : Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems
# @paper link : https://arxiv.org/abs/1512.08756

from keras import backend as K, initializers, constraints, regularizers
from keras.engine.topology import Layer


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 use_bias=True,
                 return_attention_score=False,
                 **kwargs):
        """
        implements an Attention mechanism for temporal data

        # Input shape
            3D tensor with shape `(batch_size, step_size, input_features)`
        # Output shape
            2D tensor with shape `(batch_size, input_features)`
        # Example
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # Dense Layer for classification

            # 2 Get attention scores
            x = LSTM(64, return_sequences=True)(input_words)
            x, scores = Attention(return_attention_score=True)(x)
        """
        self.supports_masking = True
        self.return_attention_score = return_attention_score
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_bias = use_bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_shape[1],),
                                        initializer='zero',
                                        name='{}_b'.format(self.name),
                                        regularizer=self.b_regularizer,
                                        constraint=self.b_constraint)
        else:
            self.bias = None

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        eij = K.squeeze(K.dot(x, K.reshape(self.W, (-1, 1))), axis=-1)

        if self.use_bias:
            eij += self.bias

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after exp
        if mask is not None:
            a *= K.cast(mask, K.floatx())

        # add K.epsilon() to avoid gradient exploding in early training
        a /= K.cast(K.sum(a, axis=1, keepdims=True)+K.epsilon(), K.floatx())

        weighted_input = x*K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention_score:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention_score:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]
