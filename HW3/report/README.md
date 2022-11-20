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

1. get each check point folder in train process's output_dir
2. use each check point to generate prediction
3. evaluate each check point prediction result by rouge
4. use this code to plot it
   ```
   def rouge_data(which_rouge,data):
         return {
               'rouge_r':list(reversed([epoch[which_rouge]['r'] for epoch in data])),
               'rouge_p':list(reversed([epoch[which_rouge]['p'] for epoch in data])),
               'rouge_f':list(reversed([epoch[which_rouge]['f'] for epoch in data])),
         }
   import matplotlib.pyplot as plt
   x = [0,1,2,3,4,5,6,7]
   plt.title("rouge-l curve for 7 epoch") # title
   plt.ylabel("value") # y label
   plt.xlabel("epoch") # x label
   plt.plot(x,rouge_data('rouge-l',data)['rouge_r'],label='rouge_r')
   plt.plot(x,rouge_data('rouge-l',data)['rouge_p'],label='rouge_p')
   plt.plot(x,rouge_data('rouge-l',data)['rouge_f'],label='rouge_f')
   plt.legend()
   plt.show()
   ```

## Bonus
