# How to use it

## Environment

1. 安裝 anaconda 環境
2. use `conda create --name <env> --file requirement.txt` to create env for anaconda
3. use `conda activate <env>` to start environment for anaconda
4. 加載創建環境時無法一同下載的所有相依組建
   ex: `conda install pytorch torchvision torchaudio cpuonly -c pytorch`
5. 注意: requirement.txt 中的 pytorch 環境為 mac 專用, 請利用步驟 4 下載 linux 環境的 pytorch

## Training

1. 執行 `python train_intent.py --recover=True --hidden_size=512 --schedule=0.5 --num_epoch=50 --batch_size=64 --lr=1e-3 --dropout=0.1 --num_layers=3`
1. 執行 `python train_slot.py --recover=True --hidden_size=512 --schedule=0.5 --num_epoch=200 --batch_size=64 --lr=1e-3 --dropout=0.1 --num_layers=4`

note: 若想要從頭訓練, 則須主動去掉 `recover` 參數
note: 確保已經跑了 `bash ./download.sh`

## Testing

1. 執行 `python test_intent.py --test_file '/{test file path}/test.json' --ckpt_path /{module path}/ --pred_file '/{predict result csv place}/' --num_layers=3 --hidden_size=512`
2. 執行 `python test_slot.py --test_file '/{test file path}/test.json' --ckpt_path /{module path}/ --pred_file '/{predict result csv place}/' --num_layers=4 --hidden_size=512`

note: 確保已經跑了 `bash ./download.sh`

## Other method

1. 畫 solt 結果分析圖 `python try_seqeval.py --test_file '/{test file path}/eval.json' --ckpt_path /{module path}/best.pt --num_layers=4 --hidden_size=512`

## 完整流程

sample

```
bash ./download.sh
bash ./intent_cls.sh ../data/intent/test.json ./pred.intent.csv
bash ./slot_tag.sh ../data/slot/test.json ./pred.slot.csv
```
