import os
import json
import numpy as np

# 초성 리스트. 00 ~ 18
first_sound_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
# 중성 리스트. 00 ~ 20
middle_sound_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
# 종성 리스트. 00 ~ 27 + 1(1개 없음)
last_sound_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
symbol_list = ['`', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+', '\\', '|', '[', ']', '{', '}', ';', ':', '"', "'", ',', '.', '<', '>', '?', '/']
english_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

special_tokens = ['<syllable_pad>', '<morpheme_pad>', '<unk>']


# parameters
vector_dim = 300

# file paths
here = os.path.dirname(os.path.abspath(__file__))
file_path_tokens_map = os.path.join(here, 'data', 'tokens_map.json')
file_path_vectors_map = os.path.join(here, 'data', 'vectors_map.txt')


def normalized_random_vector(dim=300):
    vector = np.random.normal(0, 0.1, dim)
    norm_vector = vector / np.linalg.norm(vector)
    return ' '.join(str(e) for e in norm_vector)


# def crawler(url, selector):
#     req = requests.get(url)
#     html = req.text
#     soup = bs(html, 'html.parser')
#     elements = soup.select(selector)
#     for element in elements:
#         print(element.text)


if __name__ == '__main__':
    token_list = first_sound_list + middle_sound_list + last_sound_list + number_list + symbol_list + english_list + special_tokens
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

    # crawler('https://www.topikguide.com/6000-most-common-korean-words-1/', selector='body > div.site-container > div.site-inner > div > div > main > article > div > table > tbody > tr > td:nth-child(2) > p')

    # with open(file_path_vocabs_map, 'r') as i:
    #     a = json.loads(i.read())
    #     print(a)

    # with open(file_path_vectors_map, 'r') as i:
    #     lines = i.readlines()
