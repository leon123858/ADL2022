# HW2

## Set environment

create python 3.9 (by anaconda)
`conda create --name <env name> python=3.9`
you can create python environment by any other method

install dependence TA allowed

- PyTorch 1.12.1, TensorFlow 2.10.0
- Tqdm,numpy, pandas, scikit-learn 1.1.2, nltk 3.7
- transformers==4.22.2, datasets==2.5.2, accelerate==0.13.0
- gdown

## Train

should use this command before below command
`bash download.sh && python preprocess.py`

### for MC

use this command (result in output_dir)
`python train_select.py --max_seq_length=512 --pad_to_max_length --train_file='<file path>' --validation_file='<file path>' --model_name_or_path ckiplab/bert-base-chinese --tokenizer_name bert-base-chinese --do_train --do_eval --learning_rate 5e-5 --num_train_epochs 1 --output_dir ./tmp --per_gpu_eval_batch_size=2 --per_device_train_batch_size=2 --gradient_accumulation_steps=2 --overwrite_output`

or use another model (result in output_dir)

`python train_select.py --max_seq_length=512 --pad_to_max_length --train_file='<file path>' --validation_file='<file path>' --model_name_or_path ckiplab/albert-tiny-chinese --tokenizer_name bert-base-chinese --do_train --do_eval --learning_rate 5e-5 --num_train_epochs 3 --output_dir ./tmp --per_gpu_eval_batch_size=8 --per_device_train_batch_size=8 --gradient_accumulation_steps=2 --overwrite_output`

### for QA

use this command (result in output_dir)
`python run_qa.py --max_seq_length=512 --pad_to_max_length --train_file='<file path>' --validation_file='<file path>' --model_name_or_path hfl/chinese-roberta-wwm-ext --tokenizer_name bert-base-chinese --do_train --do_eval --learning_rate 3e-5 --num_train_epochs 6 --output_dir ./tmp --per_device_train_batch_size=8 --doc_stride 128 --gradient_accumulation_steps=2`

## Predict

use this command
`bash download.sh && bash run.sh '<path/to/context.json>' '<path/to/test.json>' '<path/to/prediction/result.csv>'`

## Plot

### Loss

1. get `trainer_state.json` file in output_dir,
2. get the loss in each step
3. use excel to plot it

### EM

1. set script to evaluate metric every steps
2. get EM in metric
3. use excel to plot it

## Bonus (train)

### intent

1. upload `bonus/train_intent.ipynb` to google colab
2. upload data by zip
3. execute all code

### slot

use this command (should download some dependent package)
`python train_slot.py --text_column_name 'tokens' --label_column_name 'tags' --model_name_or_path bert-base-uncased --train_file '<train file>' --validation_file '<valid file>' --output_dir ./tmp --do_train --do_eval`
