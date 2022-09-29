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
        dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False, drop_last=False, num_workers=1)
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
    with torch.no_grad():
        model.eval()
        epoch_acc = 0
        item_count = 0
        for i, batch in enumerate(data_loader):
            output, hidden = model(batch, hidden)
            hidden = hidden.detach()
            clone_batch = batch['target'].clone().to('cpu')
            _, dataIndex = output.topk(1)
            _, targetIndex = clone_batch.topk(1)
            for i in range(0, args.batch_size):
                item_index = dataIndex[i][0]
                ans_index = targetIndex[i][0]
                epoch_acc += 1 if torch.eq(item_index,
                                           ans_index) == torch.tensor(True) else 0
                item_count += 1
        print('acc:', epoch_acc/item_count)

    # TODO: write prediction to file (args.pred_file)


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
