import torch
import torch.nn as nn


class SyllableLayer(nn.Module):
    def __init__(self, type, num_layers, input_size=3, embedding_size=300, output_size=1, max_len_morpheme=5):
        super(SyllableLayer, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.layers = {}
        if type == 'linear':
            i = 0
            for n in range(num_layers-1):
                self.layers[n] = nn.linear(in_features=input_size, out_features=input_size)
                i = i + 1
            self.layers[i] = nn.linear(in_features=input_size, out_features=output_size)
        elif type == 'lstm':
            self.layers[0] = nn.LSTM(input_size=input_size, hidden_size=output_size, num_layers=num_layers)
            self.h0 = torch.randn(num_layers, max_len_morpheme, output_size)
            self.c0 = torch.randn(num_layers, max_len_morpheme, output_size)

        self.type = type

    def forward(self, inputs):
        # inputs_size:
        return None

    def get_embedding_param(self):
        return self.embedding.weight

    def set_embedding_param(self, pretrained_weight):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
