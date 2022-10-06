import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SlotModel
from utils import Vocab

import seqeval.metrics as eval
from seqeval.scheme import IOB2


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, slot2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn_slot, shuffle=False, drop_last=False, num_workers=1)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlotModel(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()
    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))
    # TODO: predict dataset
    outputFile = {}
    targetFile = {}
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(data_loader):
            output = model(batch, False)
            _, dataIndex = output.topk(1, dim=2)
            _, targetIndex = batch['target'].clone().topk(1, dim=2)
            for i in range(0, len(batch['id'])):
                items = []
                targets = []
                count = batch['count'][i]
                for j in range(0, count):
                    item_index = dataIndex[i][j][0].item()
                    target_index = targetIndex[i][j][0].item()
                    if target_index == dataset.num_classes:
                        targets.append(-1)
                    else:
                        targets.append(target_index)
                    if item_index == dataset.num_classes:
                        items.append(-1)
                    else:
                        items.append(item_index)
                id = batch['id'][i]
                outputFile[id] = [dataset.idx2label(
                    item_index) if item_index >= 0 else "unKnow" for _, item_index in enumerate(items)]
                targetFile[id] = [dataset.idx2label(
                    target_index) if target_index >= 0 else "unKnow" for _, target_index in enumerate(targets)]
    item_num = len(outputFile)
    pred_arr = []
    target_arr = []
    for i in range(0, item_num):
        pred_arr.append(outputFile["eval-{}".format(i)])
        target_arr.append(targetFile["eval-{}".format(i)])
    accuracy = eval.accuracy_score(target_arr, pred_arr)
    precision = eval.precision_score(target_arr, pred_arr)
    recall = eval.recall_score(target_arr, pred_arr)
    f1 = eval.f1_score(target_arr, pred_arr)
    print(accuracy, precision, recall, f1)
    print(eval.classification_report(target_arr, pred_arr))
    print(eval.classification_report(
        target_arr, pred_arr, mode='strict', scheme=IOB2))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
