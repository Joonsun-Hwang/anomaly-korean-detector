import hgtk

split_token = 'ᴥ'


def korean_into_phoneme(korean_word):
    phoneme_list = []
    decomposed_text = hgtk.text.decompose(korean_word).split(split_token)
    for syllable in decomposed_text:
        if syllable:
            phoneme_list_tmp = []
            for phoneme in syllable:
                phoneme_list_tmp.append(phoneme)
            phoneme_list.append(phoneme_list_tmp)
    return phoneme_list


def phoneme_into_korean(phoneme_list):
    composing_text = ''
    for phoneme in phoneme_list:
        composing_text += ''.join(phoneme) + split_token
    return hgtk.text.compose(composing_text)


if __name__ == '__main__':
    print(korean_into_phoneme('황준선'))
    print(phoneme_into_korean([['ㅎ', 'ㅘ', 'ㅇ'], ['ㅈ', 'ㅜ', 'ㄴ'], ['ㅅ', 'ㅓ', 'ㄴ']]))
