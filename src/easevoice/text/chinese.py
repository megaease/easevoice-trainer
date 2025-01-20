import os
from .g2pw import G2PWPinyin, correct_pronunciation
import re
from jieba_fast import posseg
from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_normal, to_finals_tone3, to_initials, to_finals

from .symbols import PUNCTUATION
from .tone_sandhi import ToneSandhi
from .chinese_norm import TextNormalizer
from ...logger import logger
from ...utils.config import GlobalCFG


def init_g2pw():
    cfg = GlobalCFG()
    if cfg.is_g2pw:
        logger.info("use g2pw to do pinyin inference")
        model_dir = os.path.join(os.path.dirname(__file__), "data", "chinese", "G2PWModel")
        if not os.path.exists(cfg.bert_path):
            logger.error(f"model path {cfg.bert_path} not exists")
            raise FileNotFoundError(f"model path {cfg.bert_path} not exists, please download it before using")
        return G2PWPinyin(model_dir=model_dir, model_source=cfg.bert_path, v_to_u=False, neutral_tone_with_five=True)
    return None


G2PW = init_g2pw()
TONE_MODIFIER = ToneSandhi()
TEXT_NORMALIZER = TextNormalizer()

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

MUST_ERHUA = {
    "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿"
}

NOT_ERHUA = {
    "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
    "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
    "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
    "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
    "狗儿", "少儿"
}


def get_pinyin_to_symbol():
    current_file_path = os.path.dirname(__file__)
    pinyin_to_symbol_map = {
        line.split("\t")[0]: line.strip().split("\t")[1]
        for line in open(os.path.join(current_file_path, "data", "chinese", "opencpop-strict.txt")).readlines()
    }
    return pinyin_to_symbol_map


PINYIN_TO_SYMBOL_MAP = get_pinyin_to_symbol()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in REP_MAP.keys()))

    replaced_text = pattern.sub(lambda x: REP_MAP[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(PUNCTUATION) + r"]+", "", replaced_text
    )

    return replaced_text


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
    sentences = TEXT_NORMALIZER.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation_with_en(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


def text_normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    sentences = TEXT_NORMALIZER.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text


def _merge_erhua(initials: list[str],
                 finals: list[str],
                 word: str,
                 pos: str) -> tuple[list[str], list[str]]:
    """
    Do erhub.
    """
    # fix er1
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn == 'er1':
            finals[i] = 'er2'

    # 发音
    if word not in MUST_ERHUA and (word in NOT_ERHUA or
                                   pos in {"a", "j", "nr"}):
        return initials, finals

    # "……" 等情况直接返回
    if len(finals) != len(word):
        return initials, finals

    assert len(finals) == len(word)

    # 与前一个字发同音
    new_initials = []
    new_finals = []
    for i, phn in enumerate(finals):
        if i == len(finals) - 1 and word[i] == "儿" and phn in {
                "er2", "er5"
        } and word[-2:] not in NOT_ERHUA and new_finals:
            phn = "er" + new_finals[-1][-1]

        new_initials.append(initials[i])
        new_finals.append(phn)

    return new_initials, new_finals


def _get_initials_finals(word):
    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )

    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):
    phones_list = []
    word2ph = []
    for seg in segments:
        pinyins = []
        # Replace all English words in the sentence
        seg = re.sub("[a-zA-Z]+", "", seg)
        seg_cut = posseg.lcut(seg)
        seg_cut = TONE_MODIFIER.pre_merge_for_modify(seg_cut)  # pyright: ignore
        initials = []
        finals = []

        if not GlobalCFG().is_g2pw:
            for word, pos in seg_cut:
                if pos == "eng":
                    continue
                sub_initials, sub_finals = _get_initials_finals(word)
                sub_finals = TONE_MODIFIER.modified_tone(word, pos, sub_finals)
                # 儿化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)
                # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])
            print("pypinyin结果", initials, finals)
        else:
            # g2pw采用整句推理
            pinyins = G2PW.lazy_pinyin(seg, neutral_tone_with_five=True, style=Style.TONE3)  # pyright: ignore

            pre_word_length = 0
            for word, pos in seg_cut:
                sub_initials = []
                sub_finals = []
                now_word_length = pre_word_length + len(word)

                if pos == 'eng':
                    pre_word_length = now_word_length
                    continue

                word_pinyins = pinyins[pre_word_length:now_word_length]

                # 多音字消歧
                word_pinyins = correct_pronunciation(word, word_pinyins)

                for pinyin in word_pinyins:
                    if pinyin[0].isalpha():
                        sub_initials.append(to_initials(pinyin))
                        sub_finals.append(to_finals_tone3(pinyin, neutral_tone_with_five=True))
                    else:
                        sub_initials.append(pinyin)
                        sub_finals.append(pinyin)

                pre_word_length = now_word_length
                sub_finals = TONE_MODIFIER.modified_tone(word, pos, sub_finals)
                # 儿化
                sub_initials, sub_finals = _merge_erhua(sub_initials, sub_finals, word, pos)
                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])

        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if c == v:
                assert c in PUNCTUATION
                phone = [c]
                word2ph.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone
                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in PINYIN_TO_SYMBOL_MAP.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = PINYIN_TO_SYMBOL_MAP[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone
    return phones_list, word2ph


def g2p(text):
    pattern = r"(?<=[{0}])\s*".format("".join(PUNCTUATION))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, word2ph = _g2p(sentences)
    return phones, word2ph
