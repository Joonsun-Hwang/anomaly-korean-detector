import torch
import torch.nn as nn

from layers import SyllableLayer


class AnomalyKoreanDetector(nn.Module):
    def __init__(self, syllable_layer_type, syllable_num_layers, language, vocab_size,
                 embedding_size=300, phoneme_in_size=3, phoneme_out_size=1, max_len_morpheme=5):
        super(AnomalyKoreanDetector, self).__init__()
        self.syllable_layer = SyllableLayer(layer_type=syllable_layer_type, num_layers=syllable_num_layers,
                                            language=language, vocab_size=vocab_size, embedding_size=embedding_size,
                                            input_size=phoneme_in_size, output_size=phoneme_out_size,
                                            max_len_morpheme=max_len_morpheme)

    def forward(self, inputs):
        outputs = self.syllable_layer(inputs)  # (batch_size, len_sentence, len_morpheme, embedding_size)

        return outputs

    def get_embedding_param(self):
        return self.embedding.weight

    def set_embedding_param(self, pretrained_weight):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
