import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix

from language import Language
from dataset import KoreanDataset
from util import get_accuracy

here = os.path.dirname(os.path.abspath(__file__))
file_path_data = os.path.join(here, 'data', 'test.txt')
file_path_tokens_map = os.path.join(here, 'data', 'tokens_map.json')
file_path_vectors_map = os.path.join(here, 'data', 'vectors_map.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Data parameters
noise = True
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
syllable_num_layers = 2
syllable_layer_type = 'linear'
attention_num_layer = 2
attention_type = 'general'
morpheme_num_layers = 1
morpheme_layer_type = 'lstm'
sentence_num_layers = 1
sentence_layer_type = 'lstm'
classifier_num_layer = 4

start_epoch = 0
epochs = 1000
best_loss = 100
patience = 10  # maximum number of epochs to wait when min loss is not updated
waiting = 0  # how many times min loss has not been updated as it follows the epoch.
weight_decay_percentage = 0.9
weight_decay_per_epoch = 5  # decaying the weight if min loss is not updated within 'wait_decay_per_epoch'.
batch_size = 64
print_freq = 10000  # print training status every 100 iterations, print validation status every epoch
# checkpoint = os.path.join(here, 'BEST_checkpoint.pth')  # checkpoint path or none
checkpoint = os.path.join(here, '20190613191639', 'checkpoint.pth_best')  # checkpoint path


def main():
    global checkpoint, waiting, best_loss, start_epoch
    # Vocabulary
    if checkpoint is None:
        print("There isn't the checkpoint.")
        exit()
    language = Language(file_path_tokens_map=file_path_tokens_map, file_path_vectors_map=file_path_vectors_map)
    vocab_size = language.get_n_tokens()

    # Dataset
    korean_dataset = KoreanDataset(file_path_data=file_path_data, file_path_tokens_map=file_path_tokens_map,
                                   max_len_sentence=max_len_sentence, max_len_morpheme=max_len_morpheme,
                                   noise=noise, continuous=continuous)
    test_loader = torch.utils.data.DataLoader(korean_dataset, batch_size=batch_size,
                                              pin_memory=True, drop_last=True)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    waiting = checkpoint['waiting']
    model = checkpoint['model']
    model_optimizer = checkpoint['model_optimizer']

    model = model.to(device)

    # Loss function
    criterion_is_noise = nn.BCELoss().to(device)
    criterion_is_next = nn.BCELoss().to(device)
    with torch.no_grad():
        mean_loss = test(test_loader=test_loader,
                         model=model,
                         criterion_is_noise=criterion_is_noise,
                         criterion_is_next=criterion_is_next)

        best_loss = min(mean_loss, best_loss)
        print('BEST LOSS:', best_loss)


def test(test_loader, model, criterion_is_noise, criterion_is_next):
    model.eval()

    # if not self.noise:
    #     noise_threshold = 1
    # if noise_threshold < 0.05:
    #     noise_type = 'removing_phoneme'
    # elif noise_threshold < 0.1:
    #     noise_type = 'replacing_phoneme'
    # elif noise_threshold < 0.15:
    #     noise_type = 'removing_syllable'
    # elif noise_threshold < 0.2:
    #     noise_type = 'replacing_syllable'
    # elif noise_threshold < 0.25:
    #     noise_type = 'removing_morpheme'
    # elif noise_threshold < 0.3:
    #     noise_type = 'replacing_morpheme'
    # elif noise_threshold < 0.35:
    #     noise_type = 'removing_word_phrase'
    # elif noise_threshold < 0.4:
    #     noise_type = 'replacing_word_phrase'
    # else:
    #     noise_type = 'no'
    #
    # continuity_threshold = np.random.uniform(0, 1, 1)
    # if continuity_threshold < 0.5:
    #     continuity_type = 'no'
    # else:
    #     continuity_type = 'yes'

    losses = []
    accs_is_noise = []
    accs_is_next = []
    confusions_is_noise = {'True_no': 0,
                           'False_no': 0,
                           'True_removing_phoneme': 0,
                           'False_removing_phoneme': 0,
                           'True_replacing_phoneme': 0,
                           'False_replacing_phoneme': 0,
                           'True_removing_syllable': 0,
                           'False_removing_syllable': 0,
                           'True_replacing_syllable': 0,
                           'False_replacing_syllable': 0,
                           'True_removing_morpheme': 0,
                           'False_removing_morpheme': 0,
                           'True_replacing_morpheme': 0,
                           'False_replacing_morpheme': 0,
                           'True_removing_word_phrase': 0,
                           'False_removing_word_phrase': 0,
                           'True_replacing_word_phrase': 0,
                           'False_replacing_word_phrase': 0,
                           }
    confusions_is_next = [[0, 0], [0, 0]]

    data_size = len(test_loader.dataset)

    start = time.time()

    for i, (noise_type, continuity_type, origin_sentence, enc_sentence, mask) in enumerate(test_loader):
        if i % print_freq == 0:
            print('PROGESS: %.2f' % (i*batch_size/data_size))

        # enc_sentence: (batch_size, len_sentence, len_morpheme, len_phoneme)
        # print(num_morpheme, origin_sentence, enc_sentence.size())
        enc_sentence = enc_sentence.to(device)
        mask = mask.to(device)
        outputs_is_noise, outputs_is_next = model(enc_sentence, mask)

        if noise:
            noise_type = np.array(noise_type)
            noise_type_copy = noise_type.copy()
            noise_type[noise_type == 'no'] = 0
            noise_type[noise_type != '0'] = 1
            target_is_noise = torch.FloatTensor(noise_type.astype(float)).to(device)
            ce_is_noise = criterion_is_noise(outputs_is_noise, target_is_noise)
            acc_is_noise = get_accuracy(outputs_is_noise, target_is_noise)
            accs_is_noise.append(acc_is_noise)

            t = Variable(torch.Tensor([0.5])).to(device)
            binary_outputs = (outputs_is_noise > t).float() * 1

            for j, (y_actual, y_predict) in enumerate(zip(noise_type_copy, binary_outputs)):
                if 'no' in y_actual:
                    if y_predict:
                        confusions_is_noise['False_no'] += 1
                    else:
                        confusions_is_noise['True_no'] += 1
                else:
                    if 'phoneme' in y_actual:
                        if 'removing' in y_actual:
                            if y_predict:
                                confusions_is_noise['True_removing_phoneme'] += 1
                            else:
                                confusions_is_noise['False_removing_phoneme'] += 1
                        else:
                            if y_predict:
                                confusions_is_noise['True_replacing_phoneme'] += 1
                            else:
                                confusions_is_noise['False_replacing_phoneme'] += 1
                    elif 'syllable' in y_actual:
                        if 'removing' in y_actual:
                            if y_predict:
                                confusions_is_noise['True_removing_syllable'] += 1
                            else:
                                confusions_is_noise['False_removing_syllable'] += 1
                        else:
                            if y_predict:
                                confusions_is_noise['True_replacing_syllable'] += 1
                            else:
                                confusions_is_noise['False_replacing_syllable'] += 1
                    elif 'morpheme' in y_actual:
                        if 'removing' in y_actual:
                            if y_predict:
                                confusions_is_noise['True_removing_morpheme'] += 1
                            else:
                                confusions_is_noise['False_removing_morpheme'] += 1
                        else:
                            if y_predict:
                                confusions_is_noise['True_replacing_morpheme'] += 1
                            else:
                                confusions_is_noise['False_replacing_morpheme'] += 1
                    else:
                        if 'removing' in y_actual:
                            if y_predict:
                                confusions_is_noise['True_removing_word_phrase'] += 1
                            else:
                                confusions_is_noise['False_removing_word_phrase'] += 1
                        else:
                            if y_predict:
                                confusions_is_noise['True_replacing_word_phrase'] += 1
                            else:
                                confusions_is_noise['False_replacing_word_phrase'] += 1

        if continuous:
            continuity_type = np.array(continuity_type)
            continuity_type[continuity_type == 'no'] = 0
            continuity_type[continuity_type != '0'] = 1
            target_is_next = torch.FloatTensor(continuity_type.astype(float)).to(device)
            ce_is_next = criterion_is_next(outputs_is_next, target_is_next)
            acc_is_next = get_accuracy(outputs_is_next, target_is_next)
            accs_is_next.append(acc_is_next)

            t = Variable(torch.Tensor([0.5])).to(device)
            binary_outputs = (outputs_is_next > t).float() * 1

            confusions_is_next += confusion_matrix(continuity_type.astype(int), np.array(binary_outputs.to("cpu")).astype(int))

        if 'ce_is_noise' in locals() and 'ce_is_next' in locals():
            loss = ce_is_noise + ce_is_next
        elif 'ce_is_noise' in locals():
            loss = ce_is_noise
        elif 'ce_is_next' in locals():
            loss = ce_is_next
        else:
            raise ValueError('There is no loss')

        losses.append(loss)
    # TODO: 각각 최종 accuracy, confusion matrix,
    trace_test = '\nTest Loss {loss_avg:.4f}\t'.format(loss_avg=sum(losses)/len(losses))
    if 'acc_is_noise' in locals():
        trace_test += 'Noise Accuracy {acc_is_noise_avg:.4f}\t'.format(
            acc_is_noise_avg=sum(accs_is_noise)/len(accs_is_noise))
        print(confusions_is_noise)
    if 'acc_is_next' in locals():
        trace_test += 'Continuity Accuracy {acc_is_next_avg:.4f}\t'.format(
            acc_is_next_avg=sum(accs_is_next)/len(accs_is_next))
        print(confusions_is_next)
    trace_test += '\n'
    print(trace_test)

    return sum(losses)/len(losses)


if __name__ == '__main__':
    main()

