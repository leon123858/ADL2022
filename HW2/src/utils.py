import json
from typing import Dict, List


class CONFIG:
    def __init__(self, path='config.json'):
        with open(path) as json_file:
            config = json.load(json_file)
            self.SEP_TOKEN: str = config['SEP_TOKEN']
            self.START_TOKEN: str = config['START_TOKEN']
            self.STOP_TOKEN: str = config['STOP_TOKEN']
            self.MAX_LENGTH: int = config['MAX_LENGTH']
