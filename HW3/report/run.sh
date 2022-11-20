#!/bin/sh
# bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl

cd ./src
python ./preprocess.py -d "${1}" -t ../cache/test.json --no-ans
python ./preprocess.py -d ../data/public.jsonl -t ../cache/dev.json --ans
bash test.sh ../cache/test.json ../cache/dev.json ../cache ../model
python postprocess.py -o ../cache/test.json -t ../cache/generated_predictions.txt -d "${2}"