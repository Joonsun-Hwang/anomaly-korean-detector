import numpy as np
import hgtk
from konlpy.tag import Komoran

from preprocess import get_korean_phonemes_list, get_korean_first_sound_list, get_korean_middle_sound_list, get_korean_last_sound_list, \
    get_alphabet_list, get_number_list, get_symbol_list

split_token = 'ᴥ'
komoran = Komoran()


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


def korean_into_phoneme(text, noise):
    phoneme_list = []  # [word_phrase][morpheme][syllable][phoneme]
    for word_phrase in text.split(' '):
        morpheme_list = []
        for morpheme in komoran.morphs(word_phrase):
            morpheme_list.append(morpheme_into_phoneme(morpheme))
        phoneme_list.append(morpheme_list)
    if noise is not 'no':
        phoneme_list = add_noise(phoneme_list, noise)

    phoneme_list_without_word_phrase = []  # [morpheme][syllable][phoneme]
    for word_phrase in phoneme_list:
        if word_phrase:
            word_phrase = [x for x in word_phrase if x != []]
            word_phrase = [x for x in word_phrase if x != [[]]]
            phoneme_list_without_word_phrase += word_phrase

    return phoneme_list_without_word_phrase


def add_noise(phoneme_list, noise):
    noise_percentage = 0.25
    # phoneme_list[word_phrase][morpheme][syllable][phoneme]
    phoneme_list_with_noise = phoneme_list.copy()

    word_phrase_removing_indices = []
    morpheme_removing_indices = []
    syllable_removing_indices = []
    phoneme_removing_indices = []

    word_phrase_indices = list(range(len(phoneme_list)))
    np.random.shuffle(word_phrase_indices)
    word_phrase_split = int(np.ceil(noise_percentage * len(phoneme_list)))  # split for training and validation set
    word_phrase_noise_indices = word_phrase_indices[:word_phrase_split]
    for idx_wp, word_phrase in enumerate(phoneme_list):
        if 'word_phrase' in noise:
            if idx_wp in word_phrase_noise_indices:
                if 'removing' in noise:
                    word_phrase_removing_indices.append(idx_wp)
                else:
                    for idx_m, morpheme in enumerate(word_phrase):
                        for idx_s, syllable in enumerate(morpheme):
                            if len(syllable) == 3:
                                first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_first_sound_list()[first_sound_idx],
                                                                                 get_korean_middle_sound_list()[middle_sound_idx],
                                                                                 get_korean_last_sound_list()[last_sound_idx]]
                            elif len(syllable) == 2:
                                first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_first_sound_list()[first_sound_idx],
                                                                                 get_korean_middle_sound_list()[middle_sound_idx]]
                            elif len(syllable) == 1:
                                if syllable[0] in get_korean_phonemes_list():
                                    last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_last_sound_list()[last_sound_idx]]
                                elif syllable[0] in get_alphabet_list():
                                    list_idx = np.random.randint(low=0, high=len(get_alphabet_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_alphabet_list()[list_idx]]
                                elif syllable[0] in get_number_list():
                                    list_idx = np.random.randint(low=0, high=len(get_number_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_number_list()[list_idx]]
                                elif syllable[0] in get_symbol_list():
                                    list_idx = np.random.randint(low=0, high=len(get_symbol_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_symbol_list()[list_idx]]
                            else:
                                pass

        morpheme_indices = list(range(len(word_phrase)))
        np.random.shuffle(morpheme_indices)
        morpheme_split = int(np.ceil(noise_percentage * len(word_phrase)))  # split for training and validation set
        morpheme_noise_indices = morpheme_indices[:morpheme_split]
        for idx_m, morpheme in enumerate(word_phrase):
            if 'morpheme' in noise:
                if idx_wp in word_phrase_noise_indices and idx_m in morpheme_noise_indices:
                    if 'removing' in noise:
                        morpheme_removing_indices.append((idx_wp, idx_m))
                        # del phoneme_list_with_noise[idx_wp][idx_m]
                    else:
                        for idx_s, syllable in enumerate(morpheme):
                            if len(syllable) == 3:
                                first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_first_sound_list()[first_sound_idx],
                                                                                 get_korean_middle_sound_list()[middle_sound_idx],
                                                                                 get_korean_last_sound_list()[last_sound_idx]]
                            elif len(syllable) == 2:
                                first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_first_sound_list()[first_sound_idx],
                                                                                 get_korean_middle_sound_list()[middle_sound_idx]]
                            elif len(syllable) == 1:
                                if syllable[0] in get_korean_phonemes_list():
                                    last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_last_sound_list()[last_sound_idx]]
                                elif syllable[0] in get_alphabet_list():
                                    list_idx = np.random.randint(low=0, high=len(get_alphabet_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_alphabet_list()[list_idx]]
                                elif syllable[0] in get_number_list():
                                    list_idx = np.random.randint(low=0, high=len(get_number_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_number_list()[list_idx]]
                                elif syllable[0] in get_symbol_list():
                                    list_idx = np.random.randint(low=0, high=len(get_symbol_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_symbol_list()[list_idx]]
                            else:
                                pass

            syllable_indices = list(range(len(morpheme)))
            np.random.shuffle(syllable_indices)
            syllable_split = int(np.ceil(noise_percentage * len(morpheme)))  # split for training and validation set
            syllable_noise_indices = syllable_indices[:syllable_split]
            for idx_s, syllable in enumerate(morpheme):
                if 'syllable' in noise:
                    if idx_wp in word_phrase_noise_indices and idx_m in morpheme_noise_indices and idx_s in syllable_noise_indices:
                        if 'removing' in noise:
                            syllable_removing_indices.append((idx_wp, idx_m, idx_s))
                            # del phoneme_list_with_noise[idx_wp][idx_m][idx_s]
                        else:
                            if len(syllable) == 3:
                                first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_first_sound_list()[first_sound_idx],
                                                                                 get_korean_middle_sound_list()[middle_sound_idx],
                                                                                 get_korean_last_sound_list()[last_sound_idx]]
                            elif len(syllable) == 2:
                                first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_first_sound_list()[first_sound_idx],
                                                                                 get_korean_middle_sound_list()[middle_sound_idx]]
                            elif len(syllable) == 1:
                                if syllable[0] in get_korean_phonemes_list():
                                    last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_korean_last_sound_list()[last_sound_idx]]
                                elif syllable[0] in get_alphabet_list():
                                    list_idx = np.random.randint(low=0, high=len(get_alphabet_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_alphabet_list()[list_idx]]
                                elif syllable[0] in get_number_list():
                                    list_idx = np.random.randint(low=0, high=len(get_number_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_number_list()[list_idx]]
                                elif syllable[0] in get_symbol_list():
                                    list_idx = np.random.randint(low=0, high=len(get_symbol_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s] = [get_symbol_list()[list_idx]]
                            else:
                                pass

                phoneme_indices = list(range(len(syllable)))
                np.random.shuffle(phoneme_indices)
                phoneme_split = int(np.ceil(noise_percentage * len(syllable)))  # split for training and validation set
                phoneme_noise_indices = phoneme_indices[:phoneme_split]
                for idx_p, phoneme in enumerate(syllable):
                    if 'phoneme' in noise:
                        if idx_wp in word_phrase_noise_indices and idx_m in morpheme_noise_indices and idx_s in syllable_noise_indices and idx_p in phoneme_noise_indices:
                            if 'removing' in noise:
                                phoneme_removing_indices.append((idx_wp, idx_m, idx_s, idx_p))
                                # del phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p]
                            else:
                                if idx_p == 0:
                                    if phoneme in get_korean_phonemes_list():
                                        first_sound_idx = np.random.randint(low=1, high=len(get_korean_first_sound_list()))
                                        phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p] = get_korean_first_sound_list()[first_sound_idx]
                                    elif phoneme in get_alphabet_list():
                                        list_idx = np.random.randint(low=0, high=len(get_alphabet_list()))
                                        phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p] = get_alphabet_list()[list_idx]
                                    elif phoneme in get_number_list():
                                        list_idx = np.random.randint(low=0, high=len(get_number_list()))
                                        phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p] = get_number_list()[list_idx]
                                    elif phoneme in get_symbol_list():
                                        list_idx = np.random.randint(low=0, high=len(get_symbol_list()))
                                        phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p] = get_symbol_list()[list_idx]
                                elif idx_p == 1:
                                    middle_sound_idx = np.random.randint(low=1, high=len(get_korean_middle_sound_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p] = get_korean_middle_sound_list()[middle_sound_idx]
                                elif idx_p == 2:
                                    last_sound_idx = np.random.randint(low=1, high=len(get_korean_last_sound_list()))
                                    phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p] = get_korean_last_sound_list()[last_sound_idx]
                                else:
                                    pass

    for idx_wp in sorted(word_phrase_removing_indices, reverse=True):
        del phoneme_list_with_noise[idx_wp]
    for idx_wp, idx_m in sorted(morpheme_removing_indices, reverse=True):
        del phoneme_list_with_noise[idx_wp][idx_m]
    for idx_wp, idx_m, idx_s in sorted(syllable_removing_indices, reverse=True):
        del phoneme_list_with_noise[idx_wp][idx_m][idx_s]
    for idx_wp, idx_m, idx_s, idx_p in sorted(phoneme_removing_indices, reverse=True):
        del phoneme_list_with_noise[idx_wp][idx_m][idx_s][idx_p]

    return phoneme_list_with_noise


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during back-propagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


if __name__ == '__main__':
    noise = True
    noise_threshold = np.random.uniform(0, 1, 1)
    if not noise:
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

    noise_type = 'replacing_phoneme'
    print(noise_type)
    korean_into_phoneme('커피를 마시는 중 입니다.', noise_type)
