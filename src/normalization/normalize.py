#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os.path
import traceback

from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.utils.config import *
from src.utils.helper import random_choice
from src.easevoice.text.cleaner import clean_text
from src.utils.path import format_path


class Normalize(object):
    def __init__(self, processing_path):
        self.source_path = processing_path
        self.refinements_output_path = os.path.join(self.source_path, refinements_output, refinement_file)
        self.denoises_output_path = os.path.join(self.source_path, denoises_output)
        self.output_path = os.path.join(self.source_path, random_choice())
        os.makedirs(self.output_path, exist_ok=True)

        self.normalize_text = os.path.join(normalize_root, normalize_text)
        if not os.path.exists(self.normalize_text):
            raise FileNotFoundError(self.normalize_text)

        self.normalize_ssl = os.path.join(normalize_root, normalize_ssl)
        if not os.path.exists(self.normalize_ssl):
            raise FileNotFoundError(self.normalize_ssl)

        self.normalize_token = os.path.join(normalize_root, normalize_token)
        if not os.path.exists(self.normalize_token):
            raise FileNotFoundError(self.normalize_token)

        self.device_nums = torch.cuda.device_count()
        self.cfg = cfg

    def text(self):
        output = os.path.join(str(self.output_path), text_output_name)
        if os.path.exists(output):
            os.remove(output)
        bert_dir = os.path.join(str(self.output_path), bert_output)
        os.makedirs(bert_dir, exist_ok=True)
        device = self.cfg.device
        if device != CPU:
            device = device + ":0"
        tokenizer = AutoTokenizer.from_pretrained(self.normalize_text)
        bert_model = AutoModelForMaskedLM.from_pretrained(self.normalize_text)
        bert_model = bert_model.half().to(device) if self.cfg.is_half else bert_model.to(device)
        todo = []
        res = []
        with open(self.refinements_output_path, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        language_v1_to_language_v2 = {
            "ZH": "zh",
            "zh": "zh",
            "JP": "ja",
            "jp": "ja",
            "JA": "ja",
            "ja": "ja",
            "EN": "en",
            "en": "en",
            "En": "en",
            "KO": "ko",
            "Ko": "ko",
            "ko": "ko",
            "yue": "yue",
            "YUE": "yue",
            "Yue": "yue",
        }
        for line in lines:
            try:
                wav_name, language, text = line.split("|")
                if language in language_v1_to_language_v2.keys():
                    todo.append(
                        [wav_name, text, language_v1_to_language_v2.get(language, language)]
                    )
                else:
                    print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
            except:
                print(line, traceback.format_exc())

        self._process_text(todo, res, bert_dir, tokenizer, bert_model, device)
        opt = []
        for name, phones, word2ph, norm_text in res:
            opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
        with open(output, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")

    @staticmethod
    def _get_bert_feature(text, word2ph, tokenizer, bert_model, device):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def _process_text(self, data, res, bert_dir, tokenizer, bert_model, device):
        for name, text, lan in data:
            try:
                name = format_path(name)
                name = os.path.basename(name)
                print(name)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan
                )
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan == "zh":
                    bert_feature = self._get_bert_feature(norm_text, word2ph, tokenizer, bert_model, device)
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    torch.save(bert_feature, path_bert)
                phones = " ".join(phones)
                # res.append([name,phones])
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())
