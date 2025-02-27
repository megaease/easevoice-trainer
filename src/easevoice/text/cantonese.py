# reference: https://huggingface.co/spaces/Naozumi0512/Bert-VITS2-Cantonese-Yue/blob/main/text/chinese.py

import re
import cn2an

from pyjyutping import jyutping
from .symbols import PUNCTUATION_SET, PUNCTUATION
from .chinese_norm.text_normlization import TextNormalizer


def normalizer(x):
    return cn2an.transform(x, "an2cn")


INITIALS = [
    "aa",
    "aai",
    "aak",
    "aap",
    "aat",
    "aau",
    "ai",
    "au",
    "ap",
    "at",
    "ak",
    "a",
    "p",
    "b",
    "e",
    "ts",
    "t",
    "dz",
    "d",
    "kw",
    "k",
    "gw",
    "g",
    "f",
    "h",
    "l",
    "m",
    "ng",
    "n",
    "s",
    "y",
    "w",
    "c",
    "z",
    "j",
    "ong",
    "on",
    "ou",
    "oi",
    "ok",
    "o",
    "uk",
    "ung",
    "sp",
    "spl",
    "spn",
    "sil",
]


REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in REP_MAP.keys()))

    replaced_text = pattern.sub(lambda x: REP_MAP[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(PUNCTUATION) + r"]+", "", replaced_text
    )

    return replaced_text


def text_normalize(text):
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)
    return dest_text


def jyuping_to_initials_finals_tones(jyuping_syllables):
    initials_finals = []
    tones = []
    word2ph = []

    for syllable in jyuping_syllables:
        if syllable in PUNCTUATION:
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for punctuation
        elif syllable == "_":
            initials_finals.append(syllable)
            tones.append(0)
            word2ph.append(1)  # Add 1 for underscore
        else:
            try:
                tone = int(syllable[-1])
                syllable_without_tone = syllable[:-1]
            except ValueError:
                tone = 0
                syllable_without_tone = syllable

            for initial in INITIALS:
                if syllable_without_tone.startswith(initial):
                    if syllable_without_tone.startswith("nga"):
                        initials_finals.extend(
                            [
                                syllable_without_tone[:2],
                                syllable_without_tone[2:] or syllable_without_tone[-1],
                            ]
                        )
                        tones.extend([-1, tone])
                        word2ph.append(2)
                    else:
                        final = syllable_without_tone[len(initial):] or initial[-1]
                        initials_finals.extend([initial, final])
                        tones.extend([-1, tone])
                        word2ph.append(2)
                    break
    assert len(initials_finals) == len(tones)

    # Modified to use a consonant + vowel with tone
    phones = []
    for a, b in zip(initials_finals, tones):
        if (b not in [-1, 0]):  # Prevent adding an initial 'Y' when Cantonese and Mandarin overlap; if it's punctuation, don't add it.
            todo = "%s%s" % (a, b)
        else:
            todo = a
        if (todo not in PUNCTUATION_SET):
            todo = "Y%s" % todo
        phones.append(todo)

    return phones, word2ph


def get_jyutping(text):
    jp = jyutping.convert(text)
    for symbol in PUNCTUATION:
        jp = jp.replace(symbol, " " + symbol + " ")
    jp_array = jp.split()
    return jp_array


def g2p(text):
    jyuping = get_jyutping(text)
    phones, word2ph = jyuping_to_initials_finals_tones(jyuping)
    return phones, word2ph


if __name__ == "__main__":
    text = "佢個鋤頭太短啦。"
    text = text_normalize(text)
    phones, word2ph = g2p(text)
    print(phones, word2ph)
