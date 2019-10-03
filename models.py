import torch
import torch.nn as nn

from layers import SyllableLayer, AttentionLayer, MorphemeLayer, SentenceLayer, Classifier


class AnomalyKoreanDetector(nn.Module):
    def __init__(self, len_morpheme, len_sentence, syllable_layer_type, syllable_num_layers,
                 vocab_size, attention_num_layer,
                 morpheme_layer_type, morpheme_num_layers,
                 sentence_layer_type, sentence_num_layers,
                 classifier_num_layer,
                 embedding_size=300, phoneme_in_size=3,
                 phoneme_out_size=1, morpheme_out_size=1, sentence_out_size=1,
                 attention_type='general'):
        super(AnomalyKoreanDetector, self).__init__()
        self.syllable_layer = SyllableLayer(layer_type=syllable_layer_type, num_layers=syllable_num_layers,
                                            vocab_size=vocab_size, embedding_size=embedding_size,
                                            input_size=phoneme_in_size, output_size=phoneme_out_size)
        self.attention_layer = AttentionLayer(embedding_size=embedding_size,
                                              len_morpheme=len_morpheme,
                                              num_layers=attention_num_layer,
                                              attention_type=attention_type)
        self.morpheme_layer = MorphemeLayer(layer_type=morpheme_layer_type, num_layers=morpheme_num_layers,
                                            input_size=len_morpheme, embedding_size=embedding_size,
                                            output_size=morpheme_out_size)
        self.sentence_layer = SentenceLayer(layer_type=sentence_layer_type, num_layers=sentence_num_layers,
                                            input_size=len_sentence, embedding_size=embedding_size,
                                            output_size=sentence_out_size)
        self.classifier = Classifier(num_layers=classifier_num_layer, input_size=embedding_size, output_size=1)

        self.attention_num_layer = attention_num_layer

    def forward(self, inputs, mask):
        outputs = self.syllable_layer(inputs)  # (batch_size, len_sentence, len_morpheme, embedding_size)
        for iter_layer in range(self.attention_num_layer):
            outputs = self.attention_layer(outputs, mask, iter_layer)  # (batch_size, len_sentence, len_morpheme, embedding_size)
        outputs = self.morpheme_layer(outputs)  # (batch_size, len_sentence, embedding_size)
        outputs = self.sentence_layer(outputs)  # (batch_size, embedding_size)
        outputs_is_noise, outputs_is_next = self.classifier(outputs)

        return outputs_is_noise, outputs_is_next

    def get_embedding_param(self):
        return self.embedding.weight

    def set_embedding_param(self, pretrained_weight):
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
