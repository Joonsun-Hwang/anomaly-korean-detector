import os
import json
import numpy as np
from nltk import tokenize

import hgtk
from konlpy.tag import Komoran

split_token = 'ᴥ'
komoran = Komoran()

# 초성 리스트. 00 ~ 18
first_sound_list = ['<empty_first_sound>', 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
middle_sound_list = ['<empty_middle_sound>', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
last_sound_list = ['<empty_last_sound>', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
symbol_list = ['·', '`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '\\', '|', '[', ']', '{', '}', ';', ':', '"', "'", ',', '.', '<', '>', '?', '/', ' ']
alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

special_tokens = ['<phoneme_pad>', '<unk>']


# parameters
vector_dim = 300
max_len_sentence = 50
min_len_sentence = 10
max_len_morpheme = 5
train_ratio = 0.6

# file paths
here = os.path.dirname(os.path.abspath(__file__))
file_path_tokens_map = os.path.join(here, 'data', 'tokens_map.json')
file_path_vectors_map = os.path.join(here, 'data', 'vectors_map.txt')
file_path_ko_wiki_data = os.path.join(here, 'data', 'wikiextractor', 'text')
file_path_preprocessed_data = os.path.join(here, 'data', 'data.txt')
file_path_train_data = os.path.join(here, 'data', 'train.txt')
file_path_test_data = os.path.join(here, 'data', 'test.txt')


def normalized_random_vector(dim=300):
    vector = np.random.normal(0, 0.1, dim)
    norm_vector = vector / np.linalg.norm(vector)
    return ' '.join(str(e) for e in norm_vector)


def get_korean_phonemes_list():
    token_list = first_sound_list + middle_sound_list + last_sound_list
    return sorted(set(token_list))


def get_korean_first_sound_list():
    token_list = first_sound_list
    return sorted(set(token_list))


def get_korean_middle_sound_list():
    token_list = middle_sound_list
    return sorted(set(token_list))


def get_korean_last_sound_list():
    token_list = last_sound_list
    return sorted(set(token_list))


def get_number_list():
    token_list = number_list
    return sorted(set(token_list))


def get_alphabet_list():
    token_list = alphabet_list
    return sorted(set(token_list))


def get_symbol_list():
    token_list = symbol_list
    return sorted(set(token_list))


def init_vectors_map():
    token_list = first_sound_list + middle_sound_list + last_sound_list + number_list + symbol_list + alphabet_list + special_tokens
    token_set = sorted(set(token_list))

    i = 0
    token_dict = {}
    for token in token_set:
        token_dict[token] = i
        i = i + 1

    with open(file_path_tokens_map, 'w') as o:
        json.dump(token_dict, o)

    with open(file_path_vectors_map, 'w') as o:
        for key, value in token_dict.items():
            vector_map = str(value) + ' ' + normalized_random_vector(dim=vector_dim) + '\n'
            o.write(vector_map)


def check_appropriate_sentence(text):
    sentence_phoneme = korean_into_phoneme(text.strip())
    if len(sentence_phoneme) < max_len_sentence and len(sentence_phoneme) > min_len_sentence:
        for morpheme_phoneme in sentence_phoneme:
            if len(morpheme_phoneme) > max_len_morpheme:
                return False
        return True
    else:
        return False


def ko_wiki_pre_process(file_path_ko_wiki_data, file_path_preprocessed_data):
    with open(file_path_preprocessed_data, 'w', encoding='utf-8') as o:
        path_name_list = sorted(os.listdir(file_path_ko_wiki_data))
        for path_name in path_name_list:
            file_name_list = sorted(os.listdir(os.path.join(file_path_ko_wiki_data, path_name)))
            for file_name in file_name_list:
                print(path_name, file_name)
                with open(os.path.join(file_path_ko_wiki_data, path_name, file_name), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    previous_line = ''
                    for current_line in lines:
                        current_line = current_line.lower()
                        if '<doc' in current_line:
                            previous_line = current_line
                        elif '<doc' in previous_line or '</doc' in current_line or current_line == '':
                            previous_line = ''
                        else:
                            current_sentences = tokenize.sent_tokenize(current_line.strip())
                            current_sentences = list(filter(None, current_sentences))
                            if current_sentences:
                                if previous_line:
                                    if check_appropriate_sentence(previous_line) and \
                                            check_appropriate_sentence(current_sentences[0]):
                                        data = previous_line + ' ' + current_sentences[0] + '\n'
                                        o.write(data)
                                for i in range(len(current_sentences)-1):
                                    if check_appropriate_sentence(current_sentences[i]) and \
                                            check_appropriate_sentence(current_sentences[i+1]):
                                        data = current_sentences[i] + ' ' + current_sentences[i+1] + '\n'
                                        o.write(data)
                                previous_line = current_sentences[-1]
                            else:
                                previous_line = ''


def morpheme_into_phoneme(korean_word):
    phoneme_list = []
    decomposed_text = hgtk.text.decompose(korean_word).split(split_token)
    for syllable in decomposed_text:
        if syllable:
            phoneme_list_tmp = []
            for phoneme in syllable:
                phoneme_list_tmp.append(phoneme)
            phoneme_list.append(phoneme_list_tmp)
    return phoneme_list


def korean_into_phoneme(text):
    phoneme_list = []  # [word_phrase][morpheme][syllable][phoneme]
    for word_phrase in text.split(' '):
        morpheme_list = []
        for morpheme in komoran.morphs(word_phrase):
            morpheme_list.append(morpheme_into_phoneme(morpheme))
        phoneme_list.append(morpheme_list)

    phoneme_list_without_word_phrase = []  # [morpheme][syllable][phoneme]
    for word_phrase in phoneme_list:
        if word_phrase:
            word_phrase = [x for x in word_phrase if x != []]
            word_phrase = [x for x in word_phrase if x != [[]]]
            phoneme_list_without_word_phrase += word_phrase

    return phoneme_list_without_word_phrase


def data_into_train_test():
    with open(file_path_preprocessed_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        len_data = len(lines)
        split = int(len_data*train_ratio)
        
        with open(file_path_train_data, 'w', encoding='utf-8') as train:
            for line in lines[:split]:
                train.write(line)
        with open(file_path_test_data, 'w', encoding='utf-8') as test:
            for line in lines[split:]:
                test.write(line)


if __name__ == '__main__':
    init_vectors_map()
    # ko_wiki_pre_process(file_path_ko_wiki_data=file_path_ko_wiki_data, file_path_preprocessed_data=file_path_preprocessed_data)
    data_into_train_test()
