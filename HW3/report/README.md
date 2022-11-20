# HW3

## Set environment

create python 3.9 (by anaconda)
`conda create --name <env name> python=3.9`
you can create python environment by any other method

install dependence TA allowed

- Python 3.8 / 3.9 and Python Standard Library
- PyTorch 1.12.1, TensorFlow 2.10.0
- transformers, datasets, accelerate, sentencepiece
- rouge, spacy, nltk, ckiptagger, tqdm, pandas, jsonlines

(Dependencies of above packages/tools.)

Or you can just use `bash download.sh` to install dependence package

## Train

Should step by step use below command

1. pre-process data want to train this model
   use `python preprocess.py -d [path/to/data/want/train.jsonl] -t [path/for/next/step.json]`
   should create train.json in cache for train
   should create dev.json in cache for evaluate
2. train the model and save it
   `bash process.sh [path/to/train.json] [path/to/dev.json] [path/to/model/save] google/mt5-small 7`

## Predict

use `bash download.sh` to install dependence
use `bash run.sh [path/to/data/want/to/be/predict.jsonl] [path/to/result.jsonl]'` to get predict result

## Plot

1. get `trainer_state.json` file in train process's output_dir,
2. get the loss in each step
3. use excel to plot it

## Bonus
