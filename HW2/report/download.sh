#!/bin/sh
# command: bash ./download.sh
# source: https://drive.google.com/file/d/19S3Nk2O6X2MiuZEWuG4Onvtq53mpWLdp/view?usp=sharing
pip install -r requirements.txt
mkdir cache
python ./download.py
unzip data.zip