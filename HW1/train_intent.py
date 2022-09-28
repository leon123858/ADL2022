import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
IS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
# IS_MPS = False


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # print(datasets[TRAIN][0])
    # TODO: create DataLoader for train / dev datasets
    data_loader = DataLoader(
        datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn, shuffle=True, drop_last=True, num_workers=4)
    embeddings = torch.load(
        args.cache_dir / "embeddings.pt", map_location='cpu')
    # TODO: init model and move model to target device(cpu / gpu)
    num_class = datasets[TRAIN].num_classes
    print("num of class:", num_class)
    model = SeqClassifier(embeddings, args.hidden_size, args.num_layers,
                          args.dropout, args.bidirectional, num_class)
    if IS_MPS == True:
        device = torch.device('mps')
        model.to(device)
    # TODO: init optimizer
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    hidden = None
    # TODO: Inference on train set
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for i, batch in enumerate(data_loader):
            output, hidden = model(batch, hidden)
            hidden = hidden.detach()
            # GRAD_CLIP=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            loss = loss_fn(output, batch['target'].clone().to(
                'mps' if IS_MPS else 'cpu'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            model.eval()
            epoch_loss = 0
            epoch_acc = 0
            for i, batch in enumerate(data_loader):
                output, hidden = model(batch, hidden)
                hidden = hidden.detach()
                clone_batch = batch['target'].clone().to(
                    'mps' if IS_MPS else 'cpu')
                loss = loss_fn(output, clone_batch)
                epoch_loss += loss.item()
                # _, dataIndex = output.topk(1, dim=1)
                # _, targetIndex = clone_batch.topk(1, dim=1)
                # print(dataIndex)
                # print(targetIndex)
                # print(torch.eq(dataIndex, targetIndex))
                # raise "aaa"
                # epoch_acc += 1 if torch.eq(dataIndex, targetIndex)[0] else 0
            print('device:', IS_MPS, 'loss:', epoch_loss /
                  args.batch_size, 'acc:', epoch_acc/args.batch_size)

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
