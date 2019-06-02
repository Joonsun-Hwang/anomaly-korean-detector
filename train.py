import os

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from language import Language
from dataset import KoreanDataset
from models import AnomalyKoreanDetector

here = os.path.dirname(os.path.abspath(__file__))
file_path_data = os.path.join(here, 'data', 'toy_data.txt')
file_path_tokens_map = os.path.join(here, 'data', 'tokens_map.json')
file_path_vectors_map = os.path.join(here, 'data', 'vectors_map.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Data parameters
noise = True
continuous = True
max_len_sentence = 50
max_len_morpheme = 5
embedding_dim = 300  # dimension of substring embedding
phoneme_in_size = 3
phoneme_out_size = 1
morpheme_out_size = 1
sentence_out_size = 1

# Training parameters
random_seed = 0
validation_split = .2
shuffle_dataset = True
syllable_num_layers = 2
syllable_layer_type = 'lstm'
attention_num_layer = 1
attention_type = 'general'
morpheme_layer_type = 'lstm'
morpheme_num_layers = 2
sentence_layer_type = 'lstm'
sentence_num_layers = 2

start_epoch = 0
epochs = 1000
batch_size = 3
encoder_lr = 1e-4  # learning rate for encoder
decoder_lr = 4e-4  # learning rate for decoder
print_freq = 1  # print training status every 100 iterations, print validation status every epoch


def main():
    language = Language(file_path_tokens_map=file_path_tokens_map, file_path_vectors_map=file_path_vectors_map)
    vocab_size = language.get_n_tokens()
    print('total vocab_size:', vocab_size)

    korean_dataset = KoreanDataset(file_path_data=file_path_data, file_path_tokens_map=file_path_tokens_map,
                                   max_len_sentence=50, max_len_morpheme=5, noise=True, continuous=continuous)

    # Creating data indices for training and validation splits:
    dataset_size = len(korean_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(korean_dataset, batch_size=batch_size, sampler=train_sampler,
                                               pin_memory=True, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(korean_dataset, batch_size=batch_size, sampler=valid_sampler,
                                                    pin_memory=True, drop_last=True)

    model = AnomalyKoreanDetector(len_morpheme=max_len_morpheme,
                                  len_sentence=max_len_sentence,
                                  syllable_layer_type=syllable_layer_type,
                                  syllable_num_layers=syllable_num_layers,
                                  language=language,
                                  vocab_size=vocab_size,
                                  attention_num_layer=attention_num_layer,
                                  morpheme_layer_type=morpheme_layer_type,
                                  morpheme_num_layers=morpheme_num_layers,
                                  sentence_layer_type=sentence_layer_type,
                                  sentence_num_layers=sentence_num_layers,
                                  embedding_size=embedding_dim,
                                  phoneme_in_size=phoneme_in_size,
                                  phoneme_out_size=phoneme_out_size,
                                  morpheme_out_size=morpheme_out_size,
                                  sentence_out_size=sentence_out_size,
                                  attention_type=attention_type)
    model = model.to(device)

    for i, (noise_type, continuity_type, num_morpheme, origin_sentence, enc_sentence, mask) in enumerate(train_loader):
        # enc_sentence: (batch_size, len_sentence, len_morpheme, len_phoneme)
        # print(num_morpheme, origin_sentence, enc_sentence.size())
        enc_sentence = enc_sentence.to(device)
        mask = mask.to(device)
        outputs = model(enc_sentence, mask)
        # print(outputs.size())
        break


if __name__ == '__main__':
    main()
