# Scientific Discourse Tagger (SciDT)
LSTM based sequence labeling model for scientific discourse tagger. Read the [paper](https://arxiv.org/abs/1702.05398) for more details.

## Requirements
* Theano (tested with v0.8.0)
* Keras (tested with v0.3.2)
* Pretrained word embedding (recommended: http://bio.nlplab.org/#word-vectors): SciDT expects a gzipped embedding file with each line containing word and a the vector (list of floats) separated by spaces

## Input Format
SciDT expects inputs to lists of clauses, with paragraph boundaries identified, i.e., each line in the input file needs to be a clause and and paragraphs should be separated by blank lines.

If you are training, the file additionally needs labels at the clause level, which can be specified on each line, after the clause, separated by a tab. Please look at the sample [train](https://github.com/edvisees/sciDT/blob/master/toy_train.txt) and [test](https://github.com/edvisees/sciDT/blob/master/toy_test.txt) files for the expected format.


## Intended Usage
As mentioned in the paper, the model is intended for tagging discourse elements in experiment narratives in biomedical research papers, and we use the seven label taxonomy described in [De Waard and Pander Maat (2012)][http://www.sciencedirect.com/science/article/pii/S1475158512000471]. The taxonomy is defined at the clause level, which is why we assume that each line in the input file is a clause. However, the model itself is more general than this, and can be put to use for tagging other kinds of discourse elements as well, even at the sentence level. If you find other uses for this code, I would love to hear about it!

## Training
```
python nn_passage_tagger.py --repfile REPFILE --train_file TRAINFILE --use_attention
```
where `REPFILE` is the embedding file. `--use_attention` is recommended. Check out the help messages for `nn_passage_tagger.py` for more options

### Trained model
After you train successfully, three new files appear in the directory, with file names containing chosen values for `att`, `cont` and `bi`:
* `model_att=*_cont=*_bi=*_config.json`: The model description
* `model_att=*_cont=*_bi=*_label_ind.json`: The label index
* `model_att=*_cont=*_bi=*_weights`: Learned model weights

## Testing
You can specify test files while training itself using `--test_files` arguments. Alternatively, you can do it after training is done. In the latter case, `nn_passage_tagger` assumes the trained model files described above are present in the directory.
```
python nn_passage_tagger.py REPFILE --test_files TESTFILE1 [TESTFILE2 ..] --use_attention
```
Make sure you use the same options for attention, context and bidirectional as you used for training.
