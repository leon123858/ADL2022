#!/bin/sh
# command: bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
python train_select.py --max_seq_length=512 --pad_to_max_length --model_name_or_path='./model/basic-select' --tokenizer_name bert-base-chinese --do_predict --prediction_file "${2}" --context_file "${1}"  --output_dir ./cache --per_gpu_eval_batch_size=4 --per_device_train_batch_size=4 --overwrite_output
python middleprocess.py "${1}" "${2}"
cd answerlib
python run_qa.py --max_seq_length=512 --pad_to_max_length --model_name_or_path='../model/advance-answer' --tokenizer_name bert-base-chinese --do_predict --test_file='../cache/middle_test_result.json' --output_dir ../cache --per_gpu_eval_batch_size=4 --per_device_train_batch_size=4
cd ..
python postprocess.py "${3}"