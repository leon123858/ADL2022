#!/bin/sh
# command: bash ./test.sh /path/to/model/dir /path/to/test.json /path/to/result/dir

python ../libs/summarization/run_summarization.py \
    --model_name_or_path "${1}" \
    --do_predict \
    --test_file "${2}" \
    --source_prefix "summarize: " \
    --output_dir "${3}" \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 64