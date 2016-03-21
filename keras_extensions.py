# Simple modification to the implementation of TimeDistributedDense in keras

from keras.layers.core import MaskedLayer
from keras import backend as K
from keras import activations, initializations, regularizers, constraints

class HigherOrderTimeDistributedDense(MaskedLayer):
    '''Apply the same dense layer on all inputs over two time dimensions.
    Useful when the input to the layer is a 4D tensor.

    # Input shape
        4D tensor with shape `(nb_sample, time_dimension1, time_dimension2, input_dim)`.

    # Output shape
        4D tensor with shape `(nb_sample, time_dimension1, time_dimension2, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
            The list should have 1 element, of shape `(input_dim, output_dim)`.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    '''
    input_ndim = 4

    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 input_dim=None, input_length1=None, input_length2=None, **kwargs):
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        self.input_length1 = input_length1
        self.input_length2 = input_length2
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length1, self.input_length2, self.input_dim)
        self.input = K.placeholder(ndim=4)
        super(HigherOrderTimeDistributedDense, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[3]

        self.W = self.init((input_dim, self.output_dim))
        self.b = K.zeros((self.output_dim,))

        self.trainable_weights = [self.W, self.b]
        self.regularizers = []

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1], input_shape[2], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)

        def out_step(X_i, states):
            def in_step(x, in_states):
                output = K.dot(x, self.W) + self.b
                return output, []

            _, in_outputs, _ = K.rnn(in_step, X_i,
                                    initial_states=[],
                                    mask=None)
            return in_outputs, []
        _, outputs, _ = K.rnn(out_step, X,
                             initial_states=[],
                             mask=None)
        outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim,
                  'input_length1': self.input_length1,
                  'input_length2': self.input_length2}
        base_config = super(HigherOrderTimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
