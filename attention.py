import theano
import numpy
from theano import tensor as T

from keras import activations, initializations, regularizers
from keras.layers.core import Layer

class TensorAttention(Layer):
  '''Attention layer that operates on tensors

  '''
  input_ndim = 4
  def __init__(self, input_shape, context='word', init='glorot_uniform', activation='tanh', weights=None, **kwargs):
    self.init = initializations.get(init)
    self.activation = activations.get(activation)
    self.context = context
    self.td1, self.td2, self.wd = input_shape
    self.initial_weights = weights
    kwargs['input_shape'] = input_shape
    super(TensorAttention, self).__init__(**kwargs)

  def build(self):
    proj_dim = self.wd/2
    self.rec_hid_dim = proj_dim/2
    self.att_proj = self.init((self.wd, proj_dim))
    if self.context == 'word':
      self.att_scorer = self.init((proj_dim,))
      self.trainable_weights = [self.att_proj, self.att_scorer]
    elif self.context == 'clause':
      self.att_scorer = self.init((self.rec_hid_dim,))
      self.rec_in_weights = self.init((proj_dim, self.rec_hid_dim))
      self.rec_hid_weights = self.init((self.rec_hid_dim,self.rec_hid_dim))
      self.trainable_weights = [self.att_proj, self.att_scorer, self.rec_in_weights, self.rec_hid_weights]
    elif self.context == 'para':
      self.att_scorer = self.init((self.td1, self.td2, proj_dim))
      self.trainable_weights = [self.att_proj, self.att_scorer]
    if self.initial_weights is not None:
      self.set_weights(self.initial_weights)
      del self.initial_weights

  @property
  def output_shape(self):
    return (self.input_shape[0], self.input_shape[1], self.input_shape[3])

  def get_output(self, train=False):
    input = self.get_input(train)
    proj_input = self.activation(T.tensordot(input, self.att_proj, axes=(3,0)))
    if self.context == 'word':
      att_scores = T.tensordot(proj_input, self.att_scorer, axes=(3, 0))
    elif self.context == 'clause':
      def step(a_t, h_tm1, W_in, W, sc):
        h_t = T.tanh(T.tensordot(a_t, W_in, axes=(2,0)) + T.tensordot(h_tm1, W, axes=(2,0)))
        s_t = T.tensordot(h_t, sc, axes=(2,0))
        return h_t, s_t
      [_, scores], _ = theano.scan(step, sequences=[proj_input.dimshuffle(2,0,1,3)], outputs_info=[T.zeros((proj_input.shape[0], self.td1, self.rec_hid_dim)), None], non_sequences=[self.rec_in_weights, self.rec_hid_weights, self.att_scorer])
      att_scores = scores.dimshuffle(1,2,0)
    elif self.context == 'para':
      att_scores = T.tensordot(proj_input, self.att_scorer, axes=(3, 2)).sum(axis=(1, 2))
    # Nested scans. For shame!
    def get_sample_att(sample_input, sample_att):
      sample_att_inp, _ = theano.scan(fn=lambda s_att_i, s_input_i: T.dot(s_att_i, s_input_i), sequences=[T.nnet.softmax(sample_att), sample_input])
      return sample_att_inp

    att_input, _ = theano.scan(fn=get_sample_att, sequences=[input, att_scores])
    return att_input

  def get_config(self):
    return {'cache_enabled': True,
            'custom_name': 'tensorattention',
            'input_shape': (self.td1, self.td2, self.wd),
            'context': self.context,
            'name': 'TensorAttention',
            'trainable': True}
