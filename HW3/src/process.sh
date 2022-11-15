#!/bin/sh
# command: bash ./process.sh /path/to/train.jsonl /path/to/dev.jsonl  /path/to/dir max_len

python ../libs/summarization/run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --train_file "${1}" \
    --validation_file "${2}" \
    --source_prefix "summarize: " \
    --output_dir "${3}" \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --max_source_length 256 \
    --max_target_length 64 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2