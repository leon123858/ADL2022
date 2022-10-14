from utils import CONFIG, build_context2index
from dataset import ContextSelectionDataset
from typing import Dict, List
import json
import pickle
# load script config
config = CONFIG()
# create context2index
build_context2index()
# create datasets
with open("../data/context.json") as context_file:
    context_list: List[str] = json.load(context_file)
    with open("../data/train.json") as train_file:
        train_list: List[Dict] = json.load(train_file)
        dataset = ContextSelectionDataset(
            context_list, train_list, config.MAX_LENGTH)
        with open('../cache/train_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)
    with open("../data/valid.json") as valid_file:
        valid_list: List[Dict] = json.load(valid_file)
        dataset = ContextSelectionDataset(
            context_list, valid_list, config.MAX_LENGTH)
        with open('../cache/valid_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)
