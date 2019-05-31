import torch
from torch.utils.data import Dataset
import json
from util import korean_into_phoneme


class KoreanDataset(Dataset):
    def __init__(self, file_path_data, file_path_tokens_map,
                 max_len_sentence=50, max_len_morpheme=5,
                 noise=True):
        """
        :param file_path_data: 한국어 데이터 파일 패스
        :param file_path_tokens_map: 음소 단위 token 파일 패스
        :param max_len_sentence: 문장을 형태소로 나눴을 때의 최대 길이(즉, 한 문장에 들어갈 수 있는 형태소의 총 개수)
        :param max_len_morpheme: 형태소를 음절로 나눴을 때의 최대 길이(즉, 한 형태소에 들어갈 수 있는 음절의 총 개수)
        :param noise: 랜덤하게 노이즈를 부여할지에 대한 값 (음소, 음절, 형태소, 어절 단위로 랜덤하게 대체 혹은 삭제)
        """
        self.data = []
        with open(file_path_data, 'r') as f:
            lines = f.readlines()
            for line in lines:
                sentence_phoneme = korean_into_phoneme(line)
                if sentence_phoneme <= max_len_sentence:
                    for morpheme_phoneme in sentence_phoneme:
                        if len(morpheme_phoneme) < max_len_morpheme:
                            self.data.append(line)

        with open(file_path_tokens_map, 'r') as f:
            self.token_map = json.loads(f.read())

        self.max_len_sentence = max_len_sentence
        self.max_len_morpheme = max_len_morpheme
        self.noise = noise

    def __getitem__(self, i):
        origin_sentence = self.data[i]
        len_sentence = len(origin_sentence)
        # TODO: 음소 2개가 들어오면 종성을 padding 처리
        # TODO: 음소 1개가 들어오면 종성으로 처리 (초성과 중성을 padding 처리)
        sentence_phoneme = korean_into_phoneme(origin_sentence)
        enc_sentence = []
        for morpheme_phoneme in sentence_phoneme:
            morpheme_phoneme
        enc_word = enc_sentence + \
                   [self.token_map.get(substring, self.token_map['<unk>']) for substring in plain_sentence] + \
                   [self.token_map['<w_end>']] + [self.token_map['<c_pad>']] * (self.max_len - len(plain_sentence))
        enc_word = torch.LongTensor(enc_word)

        return origin_sentence, enc_word, len_sentence

    def __len__(self):
        return len(self.data)
