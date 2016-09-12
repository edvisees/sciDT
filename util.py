import codecs
import numpy

def read_passages(filename, is_labeled):
  str_seqs = []
  str_seq = []
  label_seqs = []
  label_seq = []
  for line in codecs.open(filename, "r", "utf-8"):
    lnstrp = line.strip()
    if lnstrp == "":
      if len(str_seq) != 0:
        str_seqs.append(str_seq)
        str_seq = []
        label_seqs.append(label_seq)
        label_seq = []
    else:
      if is_labeled:
        clause, label = lnstrp.split("\t")
        label_seq.append(label)
      else:
        clause = lnstrp
      str_seq.append(clause)
  if len(str_seq) != 0:
    str_seqs.append(str_seq)
    str_seq = []
    label_seqs.append(label_seq)
    label_seq = []
  return str_seqs, label_seqs

def evaluate(y, pred):
  accuracy = float(sum([c == p for c, p in zip(y, pred)]))/len(pred)
  num_gold = {}
  num_pred = {}
  num_correct = {}
  for c, p in zip(y, pred):
    if c in num_gold:
      num_gold[c] += 1
    else:
      num_gold[c] = 1
    if p in num_pred:
      num_pred[p] += 1
    else:
      num_pred[p] = 1
    if c == p:
      if c in num_correct:
        num_correct[c] += 1
      else:
        num_correct[c] = 1
  fscores = {}
  for p in num_pred:
    precision = float(num_correct[p]) / num_pred[p] if p in num_correct else 0.0
    recall = float(num_correct[p]) / num_gold[p] if p in num_correct else 0.0
    fscores[p] = 2 * precision * recall / (precision + recall) if precision !=0 and recall !=0 else 0.0
  weighted_fscore = sum([fscores[p] * num_gold[p] if p in num_gold else 0.0 for p in fscores]) / sum(num_gold.values())
  return accuracy, weighted_fscore, fscores

def make_folds(train_X, train_Y, num_folds):
  num_points = train_X.shape[0]
  fol_len = num_points / num_folds
  rem = num_points % num_folds
  X_folds = numpy.split(train_X, num_folds) if rem == 0 else numpy.split(train_X[:-rem], num_folds)
  Y_folds = numpy.split(train_Y, num_folds) if rem == 0 else numpy.split(train_Y[:-rem], num_folds)
  cv_folds = []
  for i in range(num_folds):
    train_folds_X = []
    train_folds_Y = []
    for j in range(num_folds):
      if i != j:
        train_folds_X.append(X_folds[j])
        train_folds_Y.append(Y_folds[j])
    train_fold_X = numpy.concatenate(train_folds_X)
    train_fold_Y = numpy.concatenate(train_folds_Y)
    cv_folds.append(((train_fold_X, train_fold_Y), (X_folds[i], Y_folds[i])))
  return cv_folds
