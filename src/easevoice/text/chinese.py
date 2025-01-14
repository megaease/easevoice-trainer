import re
from .symbols import PUNCTUATION
from .chinese_norm import TextNormalizer

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
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}


def replace_punctuation_with_en(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in REP_MAP.keys()))

    replaced_text = pattern.sub(lambda x: REP_MAP[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5A-Za-z" + "".join(PUNCTUATION) + r"]+", "", replaced_text
    )

    return replaced_text


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in PUNCTUATION)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def mix_text_normalize(text):
    # 不排除英文的文本格式化
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation_with_en(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text
