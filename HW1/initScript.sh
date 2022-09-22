#!/bin/sh
# 按照此 script 操作可以順利設置運行環境
#use 'chmod 777 <file path>' to give script permission to execute
#use 'conda create --name <env name> python=3.8' to create env for anaconda
#use 'conda activate <env name>' to get in enviroment for anaconda
conda install -c conda-forge pip-tools
pip-compile requirements.in
pip-sync requirements.txt
conda install pytorch torchvision torchaudio -c pytorch-nightly