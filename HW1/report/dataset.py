from pickletools import float8

from typing import List, Dict

import torch
from torch.utils.data import Dataset

from utils import Vocab


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

    def collate_fn_slot(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn, use as middle ware for batch
        data = [pkg['tokens'] for _, pkg in enumerate(samples)]
        encode_data = self.vocab.encode_batch(data)
        target = [[self.label_mapping[item] for _, item in enumerate(
            pkg['tags'])] for _, pkg in enumerate(samples)]
        pad_target = self.vocab.encode_batch_slot(target)
        encode_target = []
        for _, index_list in enumerate(pad_target):
            tmp = []
            for _, index in enumerate(index_list):
                element = [0]*(self.num_classes + 1)
                if index >= 0:
                    element[index] = 1
                else:
                    element[self.num_classes] = 1
                tmp.append(element)
            encode_target.append(tmp)
        data_tensor = torch.tensor(encode_data, dtype=torch.int)
        target_tensor = torch.tensor(encode_target, dtype=torch.float)
        return {
            'data': data_tensor, 'target': target_tensor
        }

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

    def collate_fn_slot_test(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn, use as middle ware for batch
        data = [pkg['tokens'] for _, pkg in enumerate(samples)]
        encode_data = self.vocab.encode_batch(data)
        data_tensor = torch.tensor(encode_data, dtype=torch.int)
        return {
            'data': data_tensor, 'id': [pkg['id'] for _, pkg in enumerate(samples)], "count": [len(pkg['tokens']) for _, pkg in enumerate(samples)]
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
