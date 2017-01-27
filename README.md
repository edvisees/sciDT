# Scientific Discourse Tagger (SciDT)
LSTM based sequence labeling model for scientific discourse tagger

## Requirements
* Theano (tested with v0.8.0)
* Keras (tested with v0.3.2)
* Pretrained word embedding (recommended: http://bio.nlplab.org/#word-vectors): SciDT expects a gzipped embedding file with each line containing word and a the vector (list of floats) separated by spaces

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
