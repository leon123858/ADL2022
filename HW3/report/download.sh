#!/bin/sh
# command: bash ./download.sh
# 下載訓練相依
cd ./libs/summarization
pip install -r requirements.txt
pip install transformers
cd ../../
cd ./libs/rouge
pip install -e tw_rouge
pip install tqdm
pip install gdown
cd ../../
# 下載數據和模型
mkdir data
mkdir model
python ./download.py
unzip -o data.zip -d ./data
unzip -o model.zip -d ./model
# 建立暫存區
mkdir cache


