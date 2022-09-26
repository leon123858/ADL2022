from typing import Dict

import torch
from torch.nn import Embedding, RNN, Linear


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        # model
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.rnn = RNN(input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                       nonlinearity='relu', dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.out_layer = Linear(hidden_size, num_class)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementError

    def forward(self, batch, hidden=None) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        data_tensor = torch.tensor(batch['data'])
        embed_out = self.embed(data_tensor)
        if hidden == None:
            out, hidden = self.rnn(embed_out)
        else:
            out, hidden = self.rnn(embed_out, hidden)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.rnn.hidden_size)
        out = self.out_layer(out)
        return out, hidden
