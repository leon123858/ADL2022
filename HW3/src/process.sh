#!/bin/sh
# command: bash ./process.sh /path/to/train.json /path/to/dev.json  /path/to/dir /path/to/base/model epoach_count

python ../libs/summarization/run_summarization.py \
    --model_name_or_path "${4}" \
    --do_train \
    --do_eval \
    --train_file "${1}" \
    --validation_file "${2}" \
    --source_prefix "summarize: " \
    --output_dir "${3}" \
    --predict_with_generate \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 64 \
    --text_column "text" \
    --summary_column "summary" \
    --adafactor 
    --num_train_epochs ${5} \
    --gradient_accumulation_steps 2