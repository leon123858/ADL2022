from typing import Dict

import torch
from torch.nn import Embedding, Linear


class SelectModel(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super(SelectModel, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)

    def forward(self, batch, IS_MPS=False) -> Dict[str, torch.Tensor]:
        data = batch['data'].clone().to('mps' if IS_MPS else 'cpu')
        # batch_size * string_len
        embed_out = self.embed(data)
        # batch_size * string_len * word_vector_len
        rnn_out, _ = self.rnn(embed_out)
        # batch_size * string_len * hidden_size * (2 if 雙向)
        encode_out = torch.mean(rnn_out, 1)
        # batch_size * hidden_size * (2 if 雙向)
        final_out = self.out_layer(encode_out)
        return final_out
