#!/bin/sh
# bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl

cd ./src
# 調整輸入數據格式
python ./preprocess.py -d ../"${1}" -t ../cache/test.json --no-ans
# 創建範例數據
python ./preprocess.py -d ../data/public.jsonl -t ../cache/dev.json --ans
# 運行測試腳本
bash test.sh ../cache/test.json ../cache/dev.json ../cache ../model
# 調整輸出格式
python postprocess.py -o ../cache/test.json -t ../cache/generated_predictions.txt -d ../"${2}"