from typing import List, Dict
from torch.utils.data import Dataset


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

    def collate_fn_train(self, samples: List[Dict]) -> Dict:
        questions = [pkg['question'] for _, pkg in enumerate(samples)]
        paragraphs = [[self.context[index]
                       for _, index in enumerate(pkg["paragraphs"])] for _, pkg in enumerate(samples)]
        target = [pkg['relevant'] for _, pkg in enumerate(samples)]
        return {
            'questions': questions, "paragraphs": paragraphs, 'target': target
        }
