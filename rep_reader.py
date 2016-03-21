import gzip
import numpy

class RepReader(object):
  def __init__(self, embedding_file):
    self.word_rep = {}
    for x in gzip.open(embedding_file):
      x_parts = x.strip().split()
      if len(x_parts) == 2:
        continue
      word = x_parts[0]
      vec = numpy.asarray([float(f) for f in x_parts[1:]])
      self.word_rep[word] = vec
    #self.word_rep = {x.split()[0]: numpy.asarray([float(f) for f in x.strip().split()[1:]]) for x in gzip.open(embedding_file)}
    self.rep_min = min([x.min() for x in self.word_rep.values()])
    self.rep_max = min([x.max() for x in self.word_rep.values()])
    self.rep_shape = self.word_rep.values()[0].shape
    self.numpy_rng = numpy.random.RandomState(12345)
  
  def get_clause_rep(self, clause):
    reps = []
    for word in clause.split():
      if word not in self.word_rep:
        rep = self.numpy_rng.uniform(low = self.rep_min, high = self.rep_max, size = self.rep_shape)
        self.word_rep[word] = rep
      else:
        rep = self.word_rep[word]
      reps.append(rep)
    return numpy.asarray(reps)
