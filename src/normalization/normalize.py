#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os.path
import traceback

from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from scipy.io import wavfile

from src.utils.config import *
from src.utils.helper import random_choice
from src.easevoice.text.cleaner import clean_text
from src.utils.path import format_path
from src.utils.response import EaseVoiceResponse, ResponseStatus
from src.utils.audio import load_audio
from src.easevoice.feature_extractor.cnhubert import CNHubert
from src.utils.helper import get_hparams_from_file
from src.easevoice.module.models import SynthesizerTrn


class Normalize(object):
    def __init__(self, processing_path: str):
        self.source_path = processing_path
        self.refinements_output_path = os.path.join(self.source_path, refinements_output, refinement_file)
        self.denoises_output_path = os.path.join(self.source_path, denoises_output)
        self.output_path = os.path.join(self.source_path, random_choice())
        os.makedirs(self.output_path, exist_ok=True)
        self.text_output_path = os.path.join(self.output_path, text_output_name)
        if os.path.exists(self.text_output_path):
            os.remove(self.text_output_path)
        self.bert_output_dir = os.path.join(self.output_path, bert_output)
        os.makedirs(self.bert_output_dir, exist_ok=True)
        self.hubert_output_dir = os.path.join(self.output_path, ssl_output)
        os.makedirs(self.hubert_output_dir, exist_ok=True)
        self.wav_output_dir = os.path.join(self.output_path, wav_output)
        os.makedirs(self.wav_output_dir, exist_ok=True)
        self.semantic_output_path = os.path.join(self.output_path, semantic_output)
        if os.path.exists(self.semantic_output_path):
            os.remove(self.semantic_output_path)

        self.normalize_text = os.path.join(normalize_root, normalize_text)
        if not os.path.exists(self.normalize_text):
            raise FileNotFoundError(self.normalize_text)

        self.normalize_ssl = os.path.join(normalize_root, normalize_ssl)
        if not os.path.exists(self.normalize_ssl):
            raise FileNotFoundError(self.normalize_ssl)

        self.normalize_token = os.path.join(normalize_root, normalize_token)
        if not os.path.exists(self.normalize_token):
            raise FileNotFoundError(self.normalize_token)

        self.cfg = cfg
        self.device = self.cfg.device
        if self.device != CPU:
            self.device = self.device + ":0"

        self.maxx = 0.95
        self.alpha = 0.5

    def text(self) -> EaseVoiceResponse:
        tokenizer = AutoTokenizer.from_pretrained(self.normalize_text)
        bert_model = AutoModelForMaskedLM.from_pretrained(self.normalize_text)
        bert_model = bert_model.half().to("cpu") if self.cfg.is_half else bert_model.to("cpu")
        todo = []
        res = []
        with open(self.refinements_output_path, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        for line in lines:
            wav_name, language, text = line.split("|")
            todo.append([wav_name, text, language])

        resp = self._process_text(todo, res, self.bert_output_dir, tokenizer, bert_model)
        if resp.status == ResponseStatus.FAILED:
            return resp
        opt = []
        for name, phones, word2ph, norm_text in res:
            opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
        with open(self.text_output_path, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "success")

    def _get_bert_feature(self, text, word2ph, tokenizer, bert_model) -> EaseVoiceResponse:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to("cpu")
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        if len(word2ph) != len(text):
            return EaseVoiceResponse(ResponseStatus.FAILED, "text and word2ph not match")

        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        data = {"phone_level_feature": phone_level_feature.T}
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "success", data)

    def _process_text(self, data, res, bert_dir, tokenizer, bert_model) -> EaseVoiceResponse:
        for name, text, lan in data:
            try:
                name = format_path(name)
                name = os.path.basename(name)
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan
                )
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan == "zh":
                    resp = self._get_bert_feature(norm_text, word2ph, tokenizer, bert_model)
                    if resp.status == ResponseStatus.FAILED:
                        return resp
                    bert_feature = resp.data["phone_level_feature"]
                    if bert_feature.shape[-1] != len(phones):
                        return EaseVoiceResponse(ResponseStatus.FAILED, "bert_feature and phones not match")
                    torch.save(bert_feature, path_bert)
                phones = " ".join(phones)
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())
                return EaseVoiceResponse(ResponseStatus.FAILED, "failed to process text")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "success")

    def ssl(self) -> EaseVoiceResponse:
        with open(self.refinements_output_path, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        failed_wavs = []
        for line in lines:
            wav_name, language, text = line.split("|")
            wav_name = format_path(wav_name)
            wav_name = os.path.basename(wav_name)
            wav_path = os.path.join(self.denoises_output_path, wav_name)
            if self._name2go(wav_name, wav_path) is False:
                failed_wavs.append((wav_name, wav_path))
        if len(failed_wavs) > 0 and self.cfg.is_half:
            self.cfg.is_half = False
            for (wav_name, wav_path) in failed_wavs:
                if self._name2go(wav_name, wav_path) is False:
                    return EaseVoiceResponse(ResponseStatus.FAILED, "failed to process wav")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "success")

    def _name2go(self, wav_name, wav_path) -> bool:
        hubert_path = os.path.join(self.hubert_output_dir, wav_name + ".pt")
        if os.path.exists(hubert_path): return True

        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.2:
            return True
        tmp_audio32 = (tmp_audio / tmp_max * (self.maxx * self.alpha * 32768)) + ((1 - self.alpha) * 32768) * tmp_audio
        tmp_audio32b = (tmp_audio / tmp_max * (self.maxx * self.alpha * 1145.14)) + ((1 - self.alpha) * 1145.14) * tmp_audio
        tmp_audio = librosa.resample(
            tmp_audio32b, orig_sr=32000, target_sr=16000
        )
        tensor_wav16 = torch.from_numpy(tmp_audio)
        tensor_wav16 = tensor_wav16.half().to("cpu") if self.cfg.is_half else tensor_wav16.to("cpu")
        cnhubert_model = CNHubert(base_path=str(self.normalize_ssl))
        model = cnhubert_model.eval()

        ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1, 2).cpu()
        if np.isnan(ssl.detach().numpy()).sum() != 0:
            return False

        wavfile.write(
            os.path.join(self.wav_output_dir, wav_name),
            32000,
            tmp_audio32.astype("int16"),
        )
        torch.save(ssl, hubert_path)
        return True

    def token(self) -> EaseVoiceResponse:
        hps = get_hparams_from_file(s2config_path)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        vq_model = vq_model.half().to("cpu") if self.cfg.is_half else vq_model.to("cpu")
        vq_model.eval()
        vq_model.load_state_dict(torch.load(str(self.normalize_token), map_location="cpu")["weight"], strict=False)
        with open(self.refinements_output_path, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        opt = ["item_name\tsemantic_audio"]
        for line in lines:
            wav_name, language, text = line.split("|")
            wav_name = format_path(wav_name)
            wav_name = os.path.basename(wav_name)
            hubert_path = os.path.join(self.hubert_output_dir, wav_name + ".pt")
            if not os.path.exists(hubert_path):
                continue
            ssl_content = torch.load(hubert_path, map_location="cpu")
            ssl_content = ssl_content.half().to("cpu") if self.cfg.is_half else ssl_content.to("cpu")
            codes = vq_model.extract_latent(ssl_content)
            semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
            opt.append("%s\t%s" % (wav_name, semantic))

        with open(self.semantic_output_path, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "success")
