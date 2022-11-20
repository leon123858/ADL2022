#!/bin/sh
# command: bash ./test.sh /path/to/test.json /path/to/dev.json  /path/to/dir /path/to/base/model

python ../libs/summarization/run_summarization.py \
    --model_name_or_path "${4}" \
    --do_predict \
    --test_file "${1}" \
    --validation_file "${2}" \
    --source_prefix "summarize: " \
    --output_dir "${3}" \
    --predict_with_generate \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 20 \
    --text_column "text" \
    --summary_column "summary" \
    --num_beams 5