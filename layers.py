import torch
import torch.nn as nn
from torch.autograd import Variable

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


class AttentionLayer(nn.Module):
    def __init__(self, embedding_size, len_morpheme, num_layers, attention_type='general'):
        super(AttentionLayer, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')
        else:
            self.attention_type = attention_type

        self.linear_in_syllable = {}
        self.linear_in_morpheme = {}
        self.linear_out_morpheme = {}

        if self.attention_type == 'general':
            for n in range(num_layers):
                self.linear_in_syllable[n] = nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=False).to(device)
                self.linear_in_morpheme[n] = nn.Linear(in_features=len_morpheme, out_features=len_morpheme, bias=False).to(device)

        for n in range(num_layers):
            self.linear_out_morpheme[n] = nn.Linear(in_features=len_morpheme*2, out_features=len_morpheme, bias=False).to(device)

        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.Tanh()

    def forward(self, inputs, mask, iter_layer):  # self-attention
        # inputs: (batch_size, len_sentence, len_morpheme, embedding_size)
        # mask: (batch_size, len_sentence, len_morpheme, 1)

        batch_size = inputs.size(0)
        len_sentence = inputs.size(1)
        len_morpheme = inputs.size(2)
        embedding_size = inputs.size(3)
        if iter_layer == 0:
            mask = torch.cat([mask]*embedding_size, dim=3)  # (batch_size, len_sentence, len_morpheme, embedding_size)

        inputs_morpheme = inputs.view(-1, len_morpheme, embedding_size)  # (batch_size * len_sentence, len_morpheme, embedding_size)
        inputs_sentence = inputs.transpose(2, 3).transpose(1, 2).contiguous()
        inputs_sentence = inputs_sentence.view(-1, len_sentence, len_morpheme)  # (batch_size * embedding_size, len_sentence, len_morpheme)
        if iter_layer == 0:
            mask_morpheme = mask.view(-1, len_morpheme, embedding_size)
            mask_sentence = mask.transpose(2, 3).transpose(1, 2).contiguous()
            mask_sentence = mask_sentence.view(-1, len_sentence, len_morpheme)

        if self.attention_type == 'general':
            inputs_morpheme = inputs_morpheme.view(-1, embedding_size)
            inputs_sentence = inputs_sentence.view(-1, len_morpheme)
            inputs_morpheme = self.linear_in_syllable[iter_layer](inputs_morpheme)
            inputs_sentence = self.linear_in_morpheme[iter_layer](inputs_sentence)
            inputs_morpheme = inputs_morpheme.view(-1, len_morpheme, embedding_size)
            inputs_sentence = inputs_sentence.view(-1, len_sentence, len_morpheme)

        attention_scores_morpheme = torch.bmm(inputs_morpheme, inputs_morpheme.transpose(1, 2).contiguous()) / math.sqrt(inputs_morpheme.size(-1))  # (batch_size*len_sentence, len_morpheme, len_morpheme)
        attention_scores_sentence = torch.bmm(inputs_sentence, inputs_sentence.transpose(1, 2).contiguous()) / math.sqrt(inputs_morpheme.size(-1))  # (batch_size*embedding_size, len_sentence, len_sentence)
        attention_scores_morpheme = Variable(attention_scores_morpheme.view(-1, len_morpheme))  # avoid inplace operation
        attention_scores_sentence = Variable(attention_scores_sentence.view(-1, len_sentence))
        if iter_layer == 0:
            attention_mask_morpheme = torch.bmm(mask_morpheme, mask_morpheme.transpose(1, 2).contiguous())
            attention_mask_sentence = torch.bmm(mask_sentence, mask_sentence.transpose(1, 2).contiguous())
            attention_mask_morpheme = attention_mask_morpheme.view(-1, len_morpheme)
            attention_mask_sentence = attention_mask_sentence.view(-1, len_sentence)
            attention_mask_morpheme = attention_mask_morpheme > 0
            attention_mask_sentence = attention_mask_sentence > 0
            attention_scores_morpheme[~attention_mask_morpheme] = float('-inf')
            attention_scores_sentence[~attention_mask_sentence] = float('-inf')

        attention_weights_morpheme = self.softmax(attention_scores_morpheme)
        attention_weights_sentence = self.softmax(attention_scores_sentence)
        attention_weights_morpheme[attention_weights_morpheme != attention_weights_morpheme] = float(0)  # nan 0으로 처리
        attention_weights_sentence[attention_weights_sentence != attention_weights_sentence] = float(0)
        attention_weights_morpheme = attention_weights_morpheme.view(-1, len_morpheme, len_morpheme)
        attention_weights_sentence = attention_weights_sentence.view(-1, len_sentence, len_sentence)

        mix = torch.bmm(attention_weights_morpheme, inputs_morpheme)  # (batch_size*len_sentence, len_morpheme, embedding_size)
        mix = mix.view(batch_size, len_sentence, len_morpheme, embedding_size)
        mix = mix.transpose(2, 3).transpose(1, 2).contiguous()
        mix = mix.view(-1, len_sentence, len_morpheme)
        mix = torch.bmm(attention_weights_sentence, mix)  # (batch_size*len_sentence, len_sentence, len_morpheme)

        combined = torch.cat((mix, inputs_sentence), dim=2)  # (batch_size*len_sentence, len_sentence, len_morpheme*2)
        combined = combined.view(-1, len_morpheme*2)

        outputs = self.linear_out_morpheme[iter_layer](combined).view(batch_size, len_sentence, len_morpheme, embedding_size)
        outputs = self.activation(outputs + inputs)

        return outputs


class SyllableLayer(nn.Module):
    def __init__(self, layer_type, num_layers, language, vocab_size,
                 embedding_size=300, input_size=3, output_size=1):
        super(SyllableLayer, self).__init__()

        if layer_type not in ['linear', 'lstm']:
            raise ValueError('Invalid syllable layer type selected.')

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.set_embedding_param(language.get_vectors())  # init embedding weight
        self.input_size = input_size

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

        self.activation = nn.ReLU()

        self.layer_type = layer_type

    def forward(self, inputs):
        # inputs: (batch_size, len_sentence, len_morpheme, len_phoneme)
        inputs = self.embedding(inputs)  # (batch_size, len_sentence, len_morpheme, len_phoneme, embedding_size)
        inputs = inputs.transpose(3, 4).contiguous()  # (batch_size, len_sentence, len_morpheme, embedding_size, len_phoneme)

        batch_size = inputs.size(0)
        len_sentence = inputs.size(1)
        len_morpheme = inputs.size(2)
        embedding_size = inputs.size(3)
        len_phoneme = inputs.size(4)

        inputs = inputs.view(-1, embedding_size, len_phoneme)
        for layer in self.layers.values():
            if hasattr(self, 'h0'):
                outputs, (hn, cn) = layer(inputs, (self.h0, self.c0))
                self.activation(outputs)
            else:
                outputs = layer(inputs)
                if outputs.size(-1) == self.input_size:
                    outputs = self.activation(outputs+inputs)
                    outputs = outputs.view(-1, embedding_size, len_phoneme)
                else:
                    outputs = self.activation(outputs)
            inputs = outputs
        outputs = inputs.view(batch_size, len_sentence, len_morpheme, embedding_size, 1).squeeze()

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


class MorphemeLayer(nn.Module):
    def __init__(self, layer_type, num_layers,
                 input_size, embedding_size=300, output_size=1):
        super(MorphemeLayer, self).__init__()

        if layer_type not in ['linear', 'lstm']:
            raise ValueError('Invalid morpheme layer type selected.')

        self.input_size = input_size
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

        self.activation = nn.ReLU()

        self.layer_type = layer_type

    def forward(self, inputs):
        # inputs: (batch_size, len_sentence, len_morpheme, embedding_size)
        inputs = inputs.transpose(2, 3).contiguous()  # (batch_size, len_sentence, embedding_size, len_morpheme)

        batch_size = inputs.size(0)
        len_sentence = inputs.size(1)
        embedding_size = inputs.size(2)
        len_morpheme = inputs.size(3)

        inputs = inputs.view(-1, embedding_size, len_morpheme)

        for layer in self.layers.values():
            if hasattr(self, 'h0'):
                outputs, (hn, cn) = layer(inputs, (self.h0, self.c0))
                self.activation(outputs)
            else:
                inputs = inputs.view(-1, len_morpheme)
                outputs = layer(inputs)
                if outputs.size(-1) == self.input_size:
                    outputs = self.activation(outputs+inputs)
                    outputs = outputs.view(-1, embedding_size, len_morpheme)
                else:
                    outputs = self.activation(outputs)
            inputs = outputs
        outputs = inputs.view(batch_size, len_sentence, embedding_size, 1).squeeze()

        return outputs

    def get_lstm_hidden0_weight(self):
        return self.h0, self.c0

    def set_lstm_hidden0_weight(self, h0, c0):
        self.h0 = h0
        self.c0 = c0


class SentenceLayer(nn.Module):
    def __init__(self, layer_type, num_layers,
                 input_size, embedding_size=300, output_size=1):
        super(SentenceLayer, self).__init__()

        if layer_type not in ['linear', 'lstm']:
            raise ValueError('Invalid morpheme layer type selected.')

        self.input_size = input_size
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

        self.activation = nn.ReLU()

        self.layer_type = layer_type

    def forward(self, inputs):
        # inputs: (batch_size, len_sentence, embedding_size)
        inputs = inputs.transpose(1, 2).contiguous()  # (batch_size, embedding_size, len_sentence)

        batch_size = inputs.size(0)
        embedding_size = inputs.size(1)
        len_sentence = inputs.size(2)

        inputs = inputs.view(-1, embedding_size, len_sentence)

        for layer in self.layers.values():
            if hasattr(self, 'h0'):
                outputs, (hn, cn) = layer(inputs, (self.h0, self.c0))
                outputs = self.activation(outputs)
            else:
                inputs = inputs.view(-1, len_sentence)
                outputs = layer(inputs)
                if outputs.size(-1) == self.input_size:
                    outputs = self.activation(outputs+inputs)
                    outputs = outputs.view(-1, embedding_size, len_sentence)
                else:
                    outputs = self.activation(outputs)
            inputs = outputs
        outputs = inputs.view(batch_size, embedding_size, 1).squeeze()

        return outputs

    def get_lstm_hidden0_weight(self):
        return self.h0, self.c0

    def set_lstm_hidden0_weight(self, h0, c0):
        self.h0 = h0
        self.c0 = c0


class Classifier(nn.Module):
    def __init__(self, num_layers,
                 input_size=300, output_size=1):
        super(Classifier, self).__init__()

        self.input_size = input_size
        self.layers_is_noise = {}
        i = 0
        for n in range(num_layers-1):
            self.layers_is_noise[n] = nn.Linear(in_features=input_size, out_features=input_size).to(device)
            i = i + 1
        self.layers_is_noise[i] = nn.Linear(in_features=input_size, out_features=output_size).to(device)

        self.layers_is_next = {}
        i = 0
        for n in range(num_layers-1):
            self.layers_is_next[n] = nn.Linear(in_features=input_size, out_features=input_size).to(device)
            i = i + 1
        self.layers_is_next[i] = nn.Linear(in_features=input_size, out_features=output_size).to(device)

        self.dropout = nn.Dropout(p=.5)
        
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # inputs: (batch_size, embedding_size)

        inputs_is_noise = inputs
        for layer in self.layers_is_noise.values():
            outputs_is_noise = layer(self.dropout(inputs_is_noise))
            if outputs_is_noise.size(-1) == self.input_size:
                outputs_is_noise = self.activation(outputs_is_noise+inputs)
            else:
                outputs_is_noise = self.activation(outputs_is_noise)
            inputs_is_noise = outputs_is_noise
        outputs_is_noise = inputs_is_noise.squeeze()
        outputs_is_noise = self.sigmoid(outputs_is_noise)

        inputs_is_next = inputs
        for layer in self.layers_is_next.values():
            outputs_is_next = layer(self.dropout(inputs_is_next))
            if outputs_is_next.size(-1) == self.input_size:
                outputs_is_next = self.activation(outputs_is_next+inputs)
            else:
                outputs_is_next = self.activation(outputs_is_next)
            inputs_is_next = outputs_is_next
        outputs_is_next = inputs_is_next.squeeze()
        outputs_is_next = self.sigmoid(outputs_is_next)

        return outputs_is_noise, outputs_is_next
