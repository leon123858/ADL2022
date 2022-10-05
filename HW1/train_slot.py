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
from model import SlotModel
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
IS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()
IS_MPS = False


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, slot2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: create DataLoader for train / dev datasets
    data_loader_train = DataLoader(
        datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn_slot, shuffle=True, drop_last=True, num_workers=4)
    data_loader_eval = DataLoader(
        datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn_slot, shuffle=True, drop_last=True, num_workers=4)
    embeddings = torch.load(
        args.cache_dir / "embeddings.pt", map_location='cpu')
    # TODO: init model and move model to target device(cpu / gpu)
    num_class = datasets[TRAIN].num_classes
    print("num of class:", num_class)
    model = SlotModel(embeddings, args.hidden_size, args.num_layers,
                      args.dropout, args.bidirectional, num_class)
    if IS_MPS == True:
        device = torch.device('mps')
        model.to(device)
    # TODO: init optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, args.schedule)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    # TODO: Inference on train set
    global_acc = 0
    before_acc_low = False
    loss_plots = []
    # when want to reload modal before
    if args.recover:
        model.load_state_dict(torch.load("./ckpt/slot/best.pt"))
    for _ in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        global_loss = 0
        model.train()
        for i, batch in enumerate(data_loader_train):
            output = model(batch, IS_MPS)
            loss = loss_fn(output, batch['target'].clone().to(
                'mps' if IS_MPS else 'cpu'))
            global_loss += loss.item()
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 2, norm_type=2)
            optimizer.step()
        # TODO: Evaluation loop - calculate accuracy and save model weights
        print("total loss:", global_loss)
        loss_plots.append(global_loss)
        with torch.no_grad():
            model.eval()
            epoch_acc = 0
            item_count = 0
            for i, batch in enumerate(data_loader_eval):
                output = model(batch, IS_MPS)
                clone_batch = batch['target'].clone().to(
                    'mps' if IS_MPS else 'cpu')
                _, dataIndex = output.topk(1, dim=2)
                _, targetIndex = clone_batch.topk(1, dim=2)
                for i in range(0, args.batch_size):
                    tmp_acc = 0
                    for j in range(0, len(dataIndex[i])):
                        item_index = dataIndex[i][j]
                        ans_index = targetIndex[i][j]
                        if torch.eq(ans_index, torch.tensor([num_class]).to('mps' if IS_MPS else 'cpu')) == torch.tensor(True).to('mps' if IS_MPS else 'cpu'):
                            tmp_acc = 1
                            break
                        if torch.eq(item_index, ans_index) == torch.tensor(False).to('mps' if IS_MPS else 'cpu'):
                            break
                    epoch_acc += tmp_acc
                    item_count += 1
            print('GPU_USED:', IS_MPS, 'acc:', epoch_acc/item_count)
            if epoch_acc > global_acc:
                global_acc = epoch_acc
                before_acc_low = False
                torch.save(model.state_dict(),
                           args.ckpt_dir / "best.pt")
            else:
                if before_acc_low == False:
                    scheduler.step()
                    print("WARNING: low the lr, learning rate is",
                          scheduler.get_last_lr(), 'now')
                before_acc_low = True

    # # TODO: Inference on test set
    print("loss_plots:", loss_plots)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
    parser.add_argument("--recover", type=bool, default=False)
    parser.add_argument("--schedule", type=float, default=0.5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(args)
    main(args)
