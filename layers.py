import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


class SyllableLayer(nn.Module):
    def __init__(self, layer_type, num_layers, language, vocab_size,
                 embedding_size=300, input_size=3, output_size=1, max_len_morpheme=5):
        super(SyllableLayer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.set_embedding_param(language.get_vectors())  # init embedding weight

        self.layers = {}
        if layer_type == 'linear':
            i = 0
            for n in range(num_layers-1):
                self.layers[n] = nn.Linear(in_features=input_size, out_features=input_size).to(device)
                i = i + 1
            self.layers[i] = nn.Linear(in_features=input_size, out_features=output_size).to(device)
        elif layer_type == 'lstm':
            self.layers[0] = nn.LSTM(input_size=input_size, hidden_size=output_size, num_layers=num_layers).to(device)
            self.h0 = torch.randn(num_layers, embedding_size, output_size).to(device)
            self.c0 = torch.randn(num_layers, embedding_size, output_size).to(device)

        self.layer_type = layer_type

    def forward(self, inputs):
        inputs = self.embedding(inputs)  # (batch_size, len_sentence, len_morpheme, len_phoneme, embedding_size)
        inputs = inputs.transpose(3, 4)  # (batch_size, len_sentence, len_morpheme, embedding_size, len_phoneme)

        batch_size = inputs.size(0)
        len_sentence = inputs.size(1)
        len_morpheme = inputs.size(2)
        embedding_size = inputs.size(3)
        len_phoneme = inputs.size(4)

        inputs = inputs.view(-1, embedding_size, len_phoneme)

        for layer in self.layers.values():
            if hasattr(self, 'h0'):
                outputs, (hn, cn) = layer(inputs, (self.h0, self.c0))
            else:
                outputs = layer(inputs)
            inputs = outputs

        outputs = inputs.squeeze()
        outputs = outputs.view(batch_size, len_sentence, len_morpheme, embedding_size)
        return outputs  # (batch_size, len_sentence, len_morpheme, embedding_size)

    def get_lstm_hidden0_weight(self):
        return self.h0, self.c0

    def set_lstm_hidden0_weight(self, h0, c0):
        self.h0 = h0
        self.c0 = c0

    def get_embedding_param(self):
        return self.embedding.weight

    def set_embedding_param(self, pretrained_weight):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
