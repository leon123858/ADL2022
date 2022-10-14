import json
from typing import Dict, List


class CONFIG:
    def __init__(self, path='config.json'):
        with open(path) as json_file:
            config = json.load(json_file)
            self.BUFFER_SIZE: int = config['BUFFER_SIZE']
            self.BATCH_SIZE: int = config['BATCH_SIZE']
            self.EPOCHS: int = config['EPOCHS']
            self.NUM_HEADS: int = config['NUM_HEADS']
            self.LEARNING_RATE: float = config['LEARNING_RATE']
            self.RECOVER: bool = config['RECOVER']
            self.DEVICE: str = config['DEVICE']
            self.START_TOKEN: str = config['START_TOKEN']
            self.STOP_TOKEN: str = config['STOP_TOKEN']
            self.MAX_LENGTH: int = config['MAX_LENGTH']


def build_context2index(source_path="../data/context.json", target_path="../cache/context2index.json"):
    with open(source_path) as json_file:
        list: List[str] = json.load(json_file)
        dict: Dict = {}
        for i, context in enumerate(list):
            dict[context] = i
        with open(target_path, 'w+', encoding='utf-8') as fp:
            json.dump(dict, fp, ensure_ascii=False)
