from typing import Dict

import torch
from torch.nn import Embedding, RNN, Linear, LSTM


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
        self.out_layer = Linear(
            hidden_size*(2 if bidirectional == True else 1), num_class)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementError

    def forward(self, batch, IS_MPS=False) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
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


class SlotModel(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SlotModel, self).__init__()
        # model
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.rnn = RNN(input_size=self.embed.embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                       nonlinearity='relu', dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.out_layer = Linear(
            hidden_size*(2 if bidirectional == True else 1), num_class + 1)

    # @property
    # def encoder_output_size(self) -> int:
    #     # TODO: calculate the output dimension of rnn
    #     raise NotImplementError

    def forward(self, batch, IS_MPS=False) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        data = batch['data'].clone().to('mps' if IS_MPS else 'cpu')
        # batch_size * string_len
        embed_out = self.embed(data)
        # batch_size * string_len * word_vector_len
        rnn_out, _ = self.rnn(embed_out)
        # batch_size * string_len * hidden_size * (2 if 雙向)
        final_out = self.out_layer(rnn_out)
        # batch_size * string_len * class_num
        return final_out
