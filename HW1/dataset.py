from typing import List, Dict

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
        self.max_len = max_len

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
        data = [pkg['text'].split() for i, pkg in enumerate(samples)]
        encode_data = self.vocab.encode_batch(data)
        target = [self.label_mapping[pkg['intent']]
                  for i, pkg in enumerate(samples)]
        encode_target = []
        for i, index in enumerate(target):
            element = [0]*self.num_classes
            element[index] = 1
            encode_target.append(element)
        return {
            'data': encode_data, 'target': encode_target
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
