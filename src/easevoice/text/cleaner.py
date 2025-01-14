import os

from . import cleaned_text_to_sequence
from .symbols import PUNCTUATION, SYMBOLS
from . import chinese
from . import japanese
from . import english
from . import korean
from . import cantonese


SPECIAL = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]

LANGUAGE_MAP = {
    "zh": chinese,
    "ja": japanese,
    "en": english,
    "ko": korean,
    "yue": cantonese,
}


def clean_text(text, language):
    symbols = SYMBOLS
    language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    if (language not in language_module_map):
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in SPECIAL:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = __import__("text."+language_module_map[language], fromlist=[language_module_map[language]])
    if hasattr(language_module, "text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text = text
    if language == "zh" or language == "yue":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ['UNK' if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    symbols = SYMBOLS
    language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean", "yue": "cantonese"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text."+language_module_map[language], fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text



if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
