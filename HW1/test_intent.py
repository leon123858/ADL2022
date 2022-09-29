import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn_test, shuffle=False, drop_last=False, num_workers=1)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
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
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(data_loader):
            output = model(batch, False)
            _, dataIndex = output.topk(1)
            for i in range(0, len(batch['id'])):
                item_index = dataIndex[i][0].item()
                intent = dataset.idx2label(item_index)
                id = batch['id'][i]
                outputFile[id] = intent
    item_num = len(outputFile)
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w+') as pred_file:
        lines = ["id,intent\n"]
        for i in range(0, item_num):
            lines.append(
                "test-{},{}\n".format(i, outputFile["test-{}".format(i)]))
        pred_file.writelines(lines)


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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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
