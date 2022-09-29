from pickletools import float8
from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab

IS_MPS = torch.backends.mps.is_available() and torch.backends.mps.is_built()


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        # string to idx
        self.label_mapping = label_mapping
        # idx to string
        self._idx2label = {idx: intent for intent,
                           idx in self.label_mapping.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn, use as middle ware for batch
        data = [pkg['text'].split() for _, pkg in enumerate(samples)]
        encode_data = self.vocab.encode_batch(data)
        target = [self.label_mapping[pkg['intent']]
                  for _, pkg in enumerate(samples)]
        encode_target = []
        for _, index in enumerate(target):
            element = [0]*self.num_classes
            element[index] = 1
            encode_target.append(element)
        data_tensor = torch.tensor(encode_data, dtype=torch.int)
        target_tensor = torch.tensor(encode_target, dtype=torch.float)
        return {
            'data': data_tensor, 'target': target_tensor
        }

    def collate_fn_test(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn, use as middle ware for batch
        data = [pkg['text'].split() for _, pkg in enumerate(samples)]
        encode_data = self.vocab.encode_batch(data)
        data_tensor = torch.tensor(encode_data, dtype=torch.int)
        return {
            'data': data_tensor, 'id': [pkg['id'] for _, pkg in enumerate(samples)]
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
