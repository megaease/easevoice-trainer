# modified from https://github.com/CjangCjengh/vits/blob/main/text/japanese.py
from .symbols import PUNCTUATION
import re
import os
import hashlib
import pyopenjtalk
from src.logger import logger

def init():
    try:
        current_file_path = os.path.dirname(__file__)

        def get_hash(fp: str) -> str:
            hash_md5 = hashlib.md5()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        base_path = os.path.join(current_file_path, "data", "japanese")
        userdict_csv = os.path.join(base_path, "userdict.csv")
        userdict_bin = os.path.join(base_path, "user.dict")
        userdict_hash = os.path.join(base_path, "userdict.md5")
        # If there is no user dictionary, generate one; if there is, check the md5, and if it is different, regenerate it
        if os.path.exists(userdict_csv):
            if not os.path.exists(userdict_bin) or get_hash(userdict_csv) != open(userdict_hash, "r", encoding='utf-8').read():
                pyopenjtalk.mecab_dict_index(userdict_csv, userdict_bin)
                with open(userdict_hash, "w", encoding='utf-8') as f:
                    f.write(get_hash(userdict_csv))

        if os.path.exists(userdict_bin):
            pyopenjtalk.update_global_jtalk_with_user_dict(userdict_bin)
    except Exception as e:
        logger.warning(f"Failed to update pyopenjtalk, err: {e}")


init()


# Regular expression matching Japanese without punctuation marks:
_JAPANESE_CHARS = re.compile(
    r"[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# Regular expression matching non-Japanese characters or punctuation marks:
_JAPANESE_MARKS = re.compile(
    r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]"
)

# List of (symbol, Japanese) pairs for marks:
_SYMBOLS_TO_JAPANESE = [(re.compile("%s" % x[0]), x[1]) for x in [("％", "パーセント")]]


# List of (consonant, sokuon) pairs:
_real_sokuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"Q([↑↓]*[kg])", r"k#\1"),
        (r"Q([↑↓]*[tdjʧ])", r"t#\1"),
        (r"Q([↑↓]*[sʃ])", r"s\1"),
        (r"Q([↑↓]*[pb])", r"p#\1"),
    ]
]

# List of (consonant, hatsuon) pairs:
_real_hatsuon = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        (r"N([↑↓]*[pbm])", r"m\1"),
        (r"N([↑↓]*[ʧʥj])", r"n^\1"),
        (r"N([↑↓]*[tdn])", r"n\1"),
        (r"N([↑↓]*[kg])", r"ŋ\1"),
    ]
]


def post_replace_ph(ph):
    rep_map = {
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
    }

    if ph in rep_map.keys():
        ph = rep_map[ph]
    return ph


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in PUNCTUATION)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def symbols_to_japanese(text):
    for regex, replacement in _SYMBOLS_TO_JAPANESE:
        text = re.sub(regex, replacement, text)
    return text


def preprocess_jap(text, with_prosody=False):
    """Reference https://r9y9.github.io/ttslearn/latest/notebooks/ch10_Recipe-Tacotron.html"""
    text = symbols_to_japanese(text)
    # English words to lower case, should have no influence on japanese words.
    text = text.lower()
    sentences = re.split(_JAPANESE_MARKS, text)
    marks = re.findall(_JAPANESE_MARKS, text)
    text = []
    for i, sentence in enumerate(sentences):
        if re.match(_JAPANESE_CHARS, sentence):
            if with_prosody:
                text += pyopenjtalk_g2p_prosody(sentence)[1:-1]
            else:
                p = pyopenjtalk.g2p(sentence)
                text += p.split(" ")

        if i < len(marks):
            if marks[i] == " ": # Prevent unexpected UNK
                continue
            text += [marks[i].replace(" ", "")]
    return text


def text_normalize(text):
    # todo: jap text normalize

    # Avoid reference leakage caused by repeated punctuation
    text = replace_consecutive_punctuation(text)
    return text

# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py


def pyopenjtalk_g2p_prosody(text, drop_unvoiced_vowels=True):
    """Extract phoneme + prosoody symbol sequence from input full-context labels.

    The algorithm is based on `Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ with some r9y9's tweaks.

    Args:
        text (str): Input text.
        drop_unvoiced_vowels (bool): whether to drop unvoiced vowels.

    Returns:
        List[str]: List of phoneme + prosody symbols.

    Examples:
        >>> from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody
        >>> pyopenjtalk_g2p_prosody("こんにちは。")
        ['^', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '$']

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104

    """
    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)

    phones = []
    for n in range(N):
        lab_curr = labels[n]

        # current phoneme
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)  # pyright: ignore
        # deal unvoiced vowels as normal vowels
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # deal with sil at the beginning and the end of text
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")

    return phones

# Copied from espnet https://github.com/espnet/espnet/blob/master/espnet2/text/phoneme_tokenizer.py


def _numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))


def g2p(norm_text, with_prosody=True):
    phones = preprocess_jap(norm_text, with_prosody)
    phones = [post_replace_ph(i) for i in phones]
    # todo: implement tones and word2ph
    return phones


if __name__ == "__main__":
    phones = g2p("Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
    print(phones)
