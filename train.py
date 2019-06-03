import os

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from language import Language
from dataset import KoreanDataset
from models import AnomalyKoreanDetector
from util import clip_gradient
from preprocess import init_vectors_map

here = os.path.dirname(os.path.abspath(__file__))
file_path_data = os.path.join(here, 'data', 'toy_data.txt')
file_path_tokens_map = os.path.join(here, 'data', 'tokens_map.json')
file_path_vectors_map = os.path.join(here, 'data', 'vectors_map.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Data parameters
noise = True
continuous = False
if continuous:
    max_len_sentence = 100 * 2
else:
    max_len_sentence = 100
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
attention_num_layer = 2
attention_type = 'general'
morpheme_layer_type = 'lstm'
morpheme_num_layers = 2
sentence_layer_type = 'lstm'
sentence_num_layers = 2
classifier_num_layer = 1

start_epoch = 0
epochs = 1000
patience = 0  # maximum number of epochs to wait when min loss is not updated
waiting = 0  # how many times min loss has not been updated as it follows the epoch.
weight_decay_percentage = 0.9
weight_decay_per_epoch = 10  #  decaying the weight if min loss is not updated within 'wait_decay_per_epoch'.
batch_size = 3
model_lr = 4e-4  # learning rate for encoder
grad_clip = 5.
print_freq = 1  # print training status every 100 iterations, print validation status every epoch
checkpoint = None  # checkpoint path or none


def main():
    # Vocabulary
    if checkpoint is None:
        init_vectors_map()
    language = Language(file_path_tokens_map=file_path_tokens_map, file_path_vectors_map=file_path_vectors_map)
    vocab_size = language.get_n_tokens()
    print('total vocab_size:', vocab_size)

    # Dataset
    korean_dataset = KoreanDataset(file_path_data=file_path_data, file_path_tokens_map=file_path_tokens_map,
                                   max_len_sentence=max_len_sentence, max_len_morpheme=max_len_morpheme,
                                   noise=noise, continuous=continuous)
    dataset_size = len(korean_dataset)
    print('total dataset_size:', dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))  # split for training and validation set

    # Model
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
                                  classifier_num_layer=classifier_num_layer,
                                  embedding_size=embedding_dim,
                                  phoneme_in_size=phoneme_in_size,
                                  phoneme_out_size=phoneme_out_size,
                                  morpheme_out_size=morpheme_out_size,
                                  sentence_out_size=sentence_out_size,
                                  attention_type=attention_type)
    model = model.to(device)

    # Optimizer
    model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_lr)

    # Loss function
    criterion_is_noise = nn.BCELoss().to(device)
    criterion_is_next = nn.BCELoss().to(device)

    for epoch in range(epochs):
        # Creating data indices for training and validation splits:
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

        for i, (noise_type, continuity_type, num_morpheme, origin_sentence, enc_sentence, mask) in enumerate(train_loader):
            # enc_sentence: (batch_size, len_sentence, len_morpheme, len_phoneme)
            # print(num_morpheme, origin_sentence, enc_sentence.size())
            enc_sentence = enc_sentence.to(device)
            mask = mask.to(device)
            outputs_is_noise, outputs_is_next = model(enc_sentence, mask)

            if noise:
                noise_type = np.array(noise_type)
                noise_type[noise_type == 'no'] = 0
                noise_type[noise_type != '0'] = 1
                target_is_noise = torch.FloatTensor(noise_type.astype(float)).to(device)
                ce_is_noise = criterion_is_noise(outputs_is_noise, target_is_noise)

            if continuous:
                continuity_type = np.array(continuity_type)
                continuity_type[continuity_type == 'no'] = 0
                continuity_type[continuity_type != '0'] = 1
                target_is_next = torch.FloatTensor(continuity_type.astype(float)).to(device)
                ce_is_next = criterion_is_next(outputs_is_next, target_is_next)

            if 'ce_is_noise' in locals() and 'ce_is_next' in locals():
                loss = ce_is_noise + ce_is_next
            elif 'ce_is_noise' in locals():
                loss = ce_is_noise
            elif 'ce_is_next' in locals():
                loss = ce_is_next
            else:
                raise ValueError('There is no loss')

            print(float(loss))

            model_optimizer.zero_grad()
            loss.backward()
            clip_gradient(model_optimizer, grad_clip)
            model_optimizer.step()

        for i, (noise_type, continuity_type, num_morpheme, origin_sentence, enc_sentence, mask) in enumerate(validation_loader):
            pass


if __name__ == '__main__':
    main()
