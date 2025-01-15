
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict, List, Tuple
import LangSegment
import torch
import re
import os
import sys
from tqdm import tqdm
from .text_segmentation import SPLITS, PUNCTUATION, split_big_text, get_split_method

from ..text import cleaned_text_to_sequence
from ..text.cleaner import clean_text
from ..text import chinese


sys.path.append(os.getcwd())


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in SPLITS) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def merge_short_text_in_array(texts: list[str], threshold: int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


class TextPreprocessor:
    def __init__(self, bert_model: AutoModelForMaskedLM,
                 tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device

    def preprocess(self, text: str, lang: str, text_split_method: str) -> List[Dict]:
        print(("############ split text ############"))
        text = self.replace_consecutive_punctuation(text)
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(("############ get bert feature ############"))
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang)
            if phones is None or norm_text == "":
                continue
            res = {
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text: str, lang: str, text_split_method: str):
        text = text.strip("\n")
        if len(text) == 0:
            return []
        if (text[0] not in SPLITS and len(get_first(text)) < 4):
            text = "。" + text if lang != "en" else "." + text
        print(("final target input:"))
        print(text)

        seg_method = get_split_method(text_split_method)
        text = seg_method(text)

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = self.filter_text(_texts)
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if not re.sub("\W+", "", text):
                # 检测一下，如果是纯符号，就跳过。
                continue
            if (text[-1] not in SPLITS):
                text += "。" if lang != "en" else "."

            # 解决句子过长导致Bert报错的问题
            if (len(text) > 510):
                texts.extend(split_big_text(text))
            else:
                texts.append(text)

        print(("final target input (after split):"))
        print(texts)
        return texts

    def segment_and_extract_feature_for_text(self, text: str, language: str) -> Tuple[list, torch.Tensor, str]:
        return self.get_phones_and_bert(text, language)

    def get_phones_and_bert(self, text: str, language: str, final: bool = False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))  # pyright: ignore
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh")
                else:
                    phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)  # pyright: ignore
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue")
            else:
                phones, word2ph, norm_text = self.clean_text_inf(formattext, language)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])  # pyright: ignore
                    textlist.append(tmp["text"])  # pyright: ignore
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":  # pyright: ignore
                        tmp["lang"] = "yue"  # pyright: ignore
                    langlist.append(tmp["lang"])  # pyright: ignore
                    textlist.append(tmp["text"])  # pyright: ignore
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":  # pyright: ignore
                        langlist.append(tmp["lang"])  # pyright: ignore
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])  # pyright: ignore
            # print(textlist)
            # print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)  # pyright: ignore
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        if not final and len(phones) < 6:  # pyright: ignore
            return self.get_phones_and_bert("." + text, language, final=True)

        return phones, bert, norm_text  # pyright: ignore

    def get_bert_feature(self, text: str, word2ph: list) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")  # pyright: ignore
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)  # pyright: ignore
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def clean_text_inf(self, text: str, language: str):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones: list, word2ph: list, norm_text: str, language: str):
        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return feature

    def filter_text(self, texts):
        _text = []
        if all(text in [None, " ", "\n", ""] for text in texts):
            raise ValueError(("请输入有效文本"))
        for text in texts:
            if text in [None, " ", ""]:
                pass
            else:
                _text.append(text)
        return _text

    def replace_consecutive_punctuation(self, text):
        punctuations = ''.join(re.escape(p) for p in PUNCTUATION)
        pattern = f'([{punctuations}])([{punctuations}])+'
        result = re.sub(pattern, r'\1', text)
        return result
