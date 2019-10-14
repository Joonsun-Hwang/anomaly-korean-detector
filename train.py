import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from language import Language
from dataset import KoreanDataset
from models import AnomalyKoreanDetector
from util import clip_gradient, save_checkpoint, adjust_learning_rate, get_accuracy
from preprocess import init_vectors_map

here = os.path.dirname(os.path.abspath(__file__))
file_path_data = os.path.join(here, 'data', 'train.txt')
file_path_tokens_map = os.path.join(here, 'data', 'tokens_map.json')
file_path_vectors_map = os.path.join(here, 'data', 'vectors_map.txt')

now = datetime.now().strftime("%Y%m%d%H%M%S")
os.mkdir(os.path.join(here, now))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Data parameters
noise = False
continuous = True
if continuous:
    max_len_sentence = 50 * 2
else:
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
syllable_num_layers = 1
syllable_layer_type = 'linear'
attention_num_layer = 4
attention_type = 'dot'
morpheme_num_layers = 2
morpheme_layer_type = 'lstm'
sentence_num_layers = 2
sentence_layer_type = 'lstm'
classifier_num_layer = 2

start_epoch = 0
epochs = int(1e+10)
best_loss = 100
patience = 10  # maximum number of epochs to wait when min loss is not updated
waiting = 0  # how many times min loss has not been updated as it follows the epoch.
weight_decay_percentage = 0.9
weight_decay_per_epoch = 5  # decaying the weight if min loss is not updated within 'wait_decay_per_epoch'.
batch_size = 128
model_lr = 4e-4  # learning rate for encoder
grad_clip = 5.
print_freq = 100  # print training status every 100 iterations, print validation status every epoch
# checkpoint = os.path.join(here, 'BEST_checkpoint.pth')  # checkpoint path or none
checkpoint = None  # checkpoint path or none


def main():
    global checkpoint, waiting, best_loss, start_epoch
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
    if checkpoint is None:
        model = AnomalyKoreanDetector(len_morpheme=max_len_morpheme,
                                      len_sentence=max_len_sentence,
                                      syllable_layer_type=syllable_layer_type,
                                      syllable_num_layers=syllable_num_layers,
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

        # Optimizer
        model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=model_lr)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        waiting = checkpoint['waiting']
        model = checkpoint['model']
        model_optimizer = checkpoint['model_optimizer']

    model = model.to(device)

    # Loss function
    criterion_is_noise = nn.BCELoss().to(device)
    criterion_is_next = nn.BCELoss().to(device)

    for epoch in range(start_epoch, epochs):
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

        if waiting >= patience:
            break
        if waiting > 0 and waiting % weight_decay_per_epoch == 0:
            adjust_learning_rate(optimizer=model_optimizer, shrink_factor=weight_decay_percentage)

        train(train_loader=train_loader,
              model=model,
              optimizer=model_optimizer,
              criterion_is_noise=criterion_is_noise,
              criterion_is_next=criterion_is_next,
              epoch=epoch)

        with torch.no_grad():
            mean_loss = validate(validation_loader=validation_loader,
                                 model=model,
                                 criterion_is_noise=criterion_is_noise,
                                 criterion_is_next=criterion_is_next)

        is_best = mean_loss < best_loss
        best_loss = min(mean_loss, best_loss)
        if not is_best:
            waiting += 1
        else:
            waiting = 0

        # Save checkpoint
        filepath = os.path.join(here, now, 'checkpoint.pth')
        save_checkpoint(filepath, epoch, waiting, model, model_optimizer, mean_loss, is_best)


def train(train_loader, model, optimizer, criterion_is_noise, criterion_is_next, epoch):
    model.train()

    losses = []
    accs_is_noise = []
    accs_is_next = []

    start = time.time()

    for i, (noise_type, continuity_type, origin_sentence, enc_sentence, mask) in enumerate(train_loader):
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
            acc_is_noise = get_accuracy(outputs_is_noise, target_is_noise)
            accs_is_noise.append(acc_is_noise)

        if continuous:
            continuity_type = np.array(continuity_type)
            continuity_type[continuity_type == 'no'] = 1
            continuity_type[continuity_type != '1'] = 0
            target_is_next = torch.FloatTensor(continuity_type.astype(float)).to(device)
            ce_is_next = criterion_is_next(outputs_is_next, target_is_next)
            acc_is_next = get_accuracy(outputs_is_next, target_is_next)
            accs_is_next.append(acc_is_next)

        if 'ce_is_noise' in locals() and 'ce_is_next' in locals():
            loss = ce_is_noise + ce_is_next
        elif 'ce_is_noise' in locals():
            loss = ce_is_noise
        elif 'ce_is_next' in locals():
            loss = ce_is_next
        else:
            raise ValueError('There is no loss')

        a = list(model.parameters())[0].clone()
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, grad_clip)
        optimizer.step()
        b = list(model.parameters())[0].clone()

        losses.append(loss)

        if i % print_freq == 0:
            trace_training = 'Epoch: [{0}][{1}/{2}]\t' \
                             'Loss {loss:.4f} ({loss_avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                         loss=loss, loss_avg=sum(losses)/len(losses))
            if 'acc_is_noise' in locals():
                trace_training += 'Noise Accuracy {acc_is_noise:.4f} ({acc_is_noise_avg:.4f})\t'.format(
                    acc_is_noise=acc_is_noise, acc_is_noise_avg=sum(accs_is_noise)/len(accs_is_noise))
            if 'acc_is_next' in locals():
                trace_training += 'Continuity Accuracy {acc_is_next:.4f} ({acc_is_next_avg:.4f})\t'.format(
                    acc_is_next=acc_is_next, acc_is_next_avg=sum(accs_is_next)/len(accs_is_next))
            # TODO: 값 추정하기
            print(trace_training, torch.equal(a, b))


def validate(validation_loader, model, criterion_is_noise, criterion_is_next):
    model.eval()

    losses = []
    accs_is_noise = []
    accs_is_next = []

    start = time.time()

    for i, (noise_type, continuity_type, origin_sentence, enc_sentence, mask) in enumerate(validation_loader):
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
            acc_is_noise = get_accuracy(outputs_is_noise, target_is_noise)
            accs_is_noise.append(acc_is_noise)

        if continuous:
            continuity_type = np.array(continuity_type)
            continuity_type[continuity_type == 'no'] = 0
            continuity_type[continuity_type != '0'] = 1
            target_is_next = torch.FloatTensor(continuity_type.astype(float)).to(device)
            ce_is_next = criterion_is_next(outputs_is_next, target_is_next)
            acc_is_next = get_accuracy(outputs_is_next, target_is_next)
            accs_is_next.append(acc_is_next)

        if 'ce_is_noise' in locals() and 'ce_is_next' in locals():
            loss = ce_is_noise + ce_is_next
        elif 'ce_is_noise' in locals():
            loss = ce_is_noise
        elif 'ce_is_next' in locals():
            loss = ce_is_next
        else:
            raise ValueError('There is no loss')

        losses.append(loss)

    trace_validation = '\nValidation Loss {loss_avg:.4f}\t'.format(loss_avg=sum(losses)/len(losses))
    if 'acc_is_noise' in locals():
        trace_validation += 'Noise Accuracy {acc_is_noise_avg:.4f}\t'.format(
            acc_is_noise_avg=sum(accs_is_noise)/len(accs_is_noise))
    if 'acc_is_next' in locals():
        trace_validation += 'Continuity Accuracy {acc_is_next_avg:.4f}\t'.format(
            acc_is_next_avg=sum(accs_is_next)/len(accs_is_next))
    trace_validation += '\n'
    print(trace_validation)

    return sum(losses)/len(losses)


if __name__ == '__main__':
    main()
