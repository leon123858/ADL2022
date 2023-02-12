# How to use it

## Environment

1. 安裝 node.js 環境
2. 在 node.js 環境安裝 yarn
3. 安裝 anaconda 環境
4. 啟動 anaconda 對應環境
5. 安裝 pytorch https://pytorch.org/
6. 安裝 tqdm seqeval

## Preprocessing

1. 執行 `yarn downloadWordModel` 下載詞嵌入用基本模型
2. 執行 `yarn preprocess`

## Training

可至 package.json 調適參數

1. 執行 `yarn trainIntent`
1. 執行 `yarn testIntent`

## Testing

可至 package.json 調適參數

1. 執行 `yarn trainSlot`
2. 執行 `yarn testSlot`

## Other method

1. 畫 solt 結果分析圖 `yarn plot`
2. 跑一次報告流程 `yarn start`
   跑完要使用 `yarn clear` 清除用完的檔案
