import hgtk
from konlpy.tag import Komoran

split_token = 'ᴥ'
komoran = Komoran()


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


def phoneme_into_korean(phoneme_list):
    composing_text = ''
    for phoneme in phoneme_list:
        composing_text += ''.join(phoneme) + split_token
    return hgtk.text.compose(composing_text)


def korean_into_word_phrase(text):
    return text.split(' ')


def word_phrase_into_morpheme(word_phrase_list):
    morpheme_list = []
    for word_phrase in word_phrase_list:
        morpheme_list.append(komoran.morphs(word_phrase))
    return morpheme_list


def morpheme_list_into_phoneme(morpheme_list):
    phoneme_list = []
    for morpheme in morpheme_list:
        morpheme_list_tmp = []
        decomposed_text = hgtk.text.decompose(morpheme).split(split_token)
        for syllable in decomposed_text:
            if syllable:
                phoneme_list_tmp = []
                for phoneme in syllable:
                    phoneme_list_tmp.append(phoneme)
                morpheme_list_tmp.append(phoneme_list_tmp)
        phoneme_list.append(morpheme_list_tmp)
    return phoneme_list


def korean_into_phoneme(text):
    phoneme_list = []
    for morpheme in komoran.morphs(text):
        phoneme_list.append(morpheme_into_phoneme(morpheme))
    return phoneme_list


if __name__ == '__main__':
    # print(morpheme_list_into_phoneme(['황', '준', '선', '이', 'ㅂ니다', '.']))
    # print(phoneme_into_korean(['황', '준', '선', '이', 'ㅂ니다', '.']))
    # print(word_phrase_into_morpheme(korean_into_word_phrase('아서 왕 전설을 배경으로 하고 있으며 7대 죄악에 모티브를 두어 제목부터가 7개의 대죄이며 주인공들이 각각의 죄악을 상징하는 기사들이다.')))
    print(len(komoran.morphs('아서 왕 전설을 배경으로 하고 있으며 7대 죄악에 모티브를 두어 제목부터가 7개의 대죄이며 주인공들이 각각의 죄악을 상징하는 기사들이다.')))
    phoneme_list = korean_into_phoneme('황준선입니다.')
    print(len(phoneme_list))
    for i in phoneme_list:  # morpheme
        print(i)
        for j in i:  # syllable
            # print(j)
            pass
