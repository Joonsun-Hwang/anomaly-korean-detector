import torch
from torch.utils.data import Dataset
import numpy as np
import json

from util import korean_into_phoneme
from preprocess import get_korean_phonemes_list, get_korean_first_sound_list, get_korean_middle_sound_list, get_korean_last_sound_list


class KoreanDataset(Dataset):
    def __init__(self, file_path_data, file_path_tokens_map,
                 max_len_sentence=50, max_len_morpheme=5,
                 noise=True, continuous=True):
        """
        :param file_path_data: 한국어 데이터 파일 패스
        :param file_path_tokens_map: 음소 단위 token 파일 패스
        :param max_len_sentence: 문장을 형태소로 나눴을 때의 최대 길이(즉, 한 문장에 들어갈 수 있는 형태소의 총 개수)
        :param max_len_morpheme: 형태소를 음절로 나눴을 때의 최대 길이(즉, 한 형태소에 들어갈 수 있는 음절의 총 개수)
        :param noise: 랜덤하게 노이즈를 부여할지에 대한 값 (음소, 음절, 형태소, 어절 단위로 랜덤하게 대체 혹은 삭제)
        """
        self.data = []
        with open(file_path_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                sentence_phoneme = korean_into_phoneme(line.strip())
                if len(sentence_phoneme) <= max_len_sentence:
                    for morpheme_phoneme in sentence_phoneme:
                        if len(morpheme_phoneme) < max_len_morpheme:
                            self.data.append(line.strip())

        self.where_continuous = 'next'  # 연속적인 문장 체크, none: 연속 문장 없음, next: 다음 문장과 연속, previous: 이전 문장과 연속

        with open(file_path_tokens_map, 'r') as f:
            self.token_map = json.loads(f.read())

        self.max_len_sentence = max_len_sentence
        self.max_len_morpheme = max_len_morpheme
        self.noise = noise
        self.continuous = continuous

    def __getitem__(self, i):
        noise_threshold = np.random.uniform(0, 1, 1)
        if not self.noise:
            noise_threshold = 1
        if noise_threshold < 0.05:
            noise_type = 'removing_phoneme'
        elif noise_threshold < 0.1:
            noise_type = 'replacing_phoneme'
        elif noise_threshold < 0.15:
            noise_type = 'removing_syllable'
        elif noise_threshold < 0.2:
            noise_type = 'replacing_syllable'
        elif noise_threshold < 0.25:
            noise_type = 'removing_morpheme'
        elif noise_threshold < 0.3:
            noise_type = 'replacing_morpheme'
        elif noise_threshold < 0.35:
            noise_type = 'removing_word_phrase'
        elif noise_threshold < 0.4:
            noise_type = 'replacing_word_phrase'
        else:
            noise_type = 'no'

        continuity_threshold = np.random.uniform(0, 1, 1)
        if continuity_threshold < 0.5:
            continuity_type = 'no'
        else:
            continuity_type = 'yes'
        # TODO: encoding 할 때 noise 반영하기
        # TODO: continuity_type 에 따라 연속 문장 반환

        origin_sentence = self.data[i]
        sentence_phoneme = korean_into_phoneme(origin_sentence)
        num_morpheme = len(sentence_phoneme)
        mask = [[[1]]] * num_morpheme  # 의미 있는 부분: 1, padding 부분: 0

        padded_sentence_phoneme = sentence_phoneme + [[[]]] * (self.max_len_sentence-num_morpheme)
        mask += [[[0]]] * (self.max_len_sentence-num_morpheme)
        enc_sentence = []
        for idx, (morpheme_phoneme, morpheme_mask) in enumerate(zip(padded_sentence_phoneme, mask)):
            padded_morpheme_phoneme = morpheme_phoneme + [[]] * (self.max_len_morpheme-len(morpheme_phoneme))
            if mask[idx] == [[1]]:
                mask[idx] = [[1]] * len(morpheme_phoneme) + [[0]] * (self.max_len_morpheme-len(morpheme_phoneme))
            else:
                mask[idx] = [[0]] * self.max_len_morpheme
            enc_morpheme = []
            for phonemes in padded_morpheme_phoneme:
                if len(phonemes) == 3:
                    enc_phonemes = [self.token_map.get(phoneme, self.token_map['<unk>']) for phoneme in phonemes]
                elif len(phonemes) == 2:  # 종성이 없는 경우
                    enc_phonemes = [self.token_map.get(phoneme, self.token_map['<unk>']) for phoneme in phonemes] + \
                                   [self.token_map['<empty_last_sound>']]
                elif len(phonemes) == 1:
                    if phonemes[0] in get_korean_phonemes_list():  # 형태소 단위 음소가 한글일 경우, 그 앞 형태소의 받침. 따라서 종성으로 처리
                        enc_phonemes = [self.token_map['<empty_first_sound>']] + [self.token_map['<empty_middle_sound>']] + \
                                       [self.token_map.get(phoneme, self.token_map['<unk>']) for phoneme in phonemes]
                    else:  # 한글이 아닐 경우, 초성으로 처리
                        enc_phonemes = [self.token_map.get(phoneme, self.token_map['<unk>']) for phoneme in phonemes] + \
                                       [self.token_map['<empty_middle_sound>']] + [self.token_map['<empty_last_sound>']]
                else:  # padding 처리
                    enc_phonemes = [self.token_map['<phoneme_pad>']] * 3
                enc_morpheme.append(enc_phonemes)
            enc_sentence.append(enc_morpheme)

        enc_sentence = torch.LongTensor(enc_sentence)  # (len_sentence, len_morpheme, len_phoneme)
        mask = torch.FloatTensor(mask)  # (len_sentence, len_morpheme, 1)

        return noise_type, continuity_type, num_morpheme, origin_sentence, enc_sentence, mask

    def __len__(self):
        return len(self.data)
