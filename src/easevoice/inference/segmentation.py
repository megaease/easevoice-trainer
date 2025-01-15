

from enum import Enum
import re
from typing import Any, Callable, Dict, Union


PUNCTUATION = set(['!', '?', '…', ',', '.', '-', " "])
SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def split_big_text(text, max_len=510):
    # 定义全角和半角标点符号
    punctuation = "".join(SPLITS)

    # 切割文本
    segments = re.split('([' + punctuation + '])', text)

    # 初始化结果列表和当前片段
    result = []
    current_segment = ''

    for segment in segments:
        # 如果当前片段加上新的片段长度超过max_len，就将当前片段加入结果列表，并重置当前片段
        if len(current_segment + segment) > max_len:
            result.append(current_segment)
            current_segment = segment
        else:
            current_segment += segment

    # 将最后一个片段加入结果列表
    if current_segment:
        result.append(current_segment)

    return result


def _split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in SPLITS:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in SPLITS:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


class SplitMethods(Enum):
    NoSplit = "no_split"
    By4Sentences = "by_4_sentences"
    By50Chars = "by_50_chars"
    ByChinesePeriod = "by_chinese_period"
    ByEnglishPeriod = "by_english_period"
    ByPunctuation = "by_punctuation"


_SPLIT_METHODS: Dict[str, Any] = dict()


def get_split_method(name: Union[SplitMethods, str]) -> Callable:
    if isinstance(name, SplitMethods):
        method = _SPLIT_METHODS.get(name.value, None)
    else:
        method = _SPLIT_METHODS.get(name, None)

    if method is None:
        raise ValueError(f"Cut method {name} not found")
    return method


def get_split_names() -> list[str]:
    return list(_SPLIT_METHODS.keys())


def _register_method(name):
    def decorator(func):
        if isinstance(name, SplitMethods):
            _SPLIT_METHODS[name.value] = func
        else:
            _SPLIT_METHODS[name] = func
        return func
    return decorator


@_register_method(SplitMethods.NoSplit)
def split_no_split(inp):
    """
    Not split
    """
    if not set(inp).issubset(PUNCTUATION):
        return inp
    else:
        return "/n"


@_register_method(SplitMethods.By4Sentences)
def split_by_4_sentences(inp):
    """
    Split the input text into 4 sentences
    """
    inp = inp.strip("\n")
    inps = _split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None  # pyright: ignore
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


@_register_method(SplitMethods.By50Chars)
def split_by_50_chars(inp):
    """
    Split the input text into 50 characters
    """
    inp = inp.strip("\n")
    inps = _split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  # 如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


@_register_method(SplitMethods.ByChinesePeriod)
def split_by_chinese_period(inp):
    """
    Split by Chinese period
    """
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


@_register_method(SplitMethods.ByEnglishPeriod)
def split_by_english_period(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(PUNCTUATION)]
    return "\n".join(opts)


@_register_method(SplitMethods.ByPunctuation)
def split_by_punctuation(inp):
    """
    Split by punctuation
    """
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


if __name__ == '__main__':
    method = get_split_method(SplitMethods.ByPunctuation)
    print(method("你好，我是小明。你好，我是小红。你好，我是小刚。你好，我是小张。"))
