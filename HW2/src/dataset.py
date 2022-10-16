from utils import CONFIG
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from torch import tensor
import torch
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
config = CONFIG()


class ContextSelectionDataset(Dataset):
    def __init__(
        self,
        context: List[str],
        data: List[Dict],
        max_tokens_len: int,
    ):
        self.context = context
        self.data = data
        self.max_tokens_len = max_tokens_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def map_train(self) -> None:
        samples = self.data
        data = [[tokenizer(pkg['question'] + config.SEP_TOKEN + self.context[index], max_length=config.MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
                 for _, index in enumerate(pkg["paragraphs"])] for _, pkg in enumerate(samples)]
        target_index = [pkg["paragraphs"].index(pkg['relevant'])
                        for _, pkg in enumerate(samples)]
        target_vector = []
        for _, index in enumerate(target_index):
            tmp = [0, 0, 0, 0]
            tmp[index] = 1
            target_vector.append(tmp)
        train_data = []
        for i, contexts in enumerate(data):
            output = {
                "contexts_input": [context.data['input_ids'] for _, context in enumerate(contexts)],
                "contexts_mask": [context.data['attention_mask'] for _, context in enumerate(contexts)],
                "target": tensor(target_vector[i], dtype=torch.int)
            }
            train_data.append(output)
        self.train_data: List[Dict] = train_data
