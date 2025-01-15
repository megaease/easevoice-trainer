#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os.path
import traceback

import torch
from faster_whisper import WhisperModel
from funasr import AutoModel
from tqdm import tqdm

from src.utils.config import asr_root, asr_fun_version
from src.utils.response import EaseVoiceResponse, ResponseStatus


class FunAsr(object):
    def __init__(self, model_size: str, language: str, precision: str):
        model_vad = os.path.join(asr_root, "speech_fsmn_vad_zh-cn-16k-common-pytorch")
        model_punc = os.path.join(asr_root, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
        self.model_vad = model_vad if os.path.exists(model_vad) else "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
        self.model_punc = model_punc if os.path.exists(model_punc) else "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
        self.vad_model_revision = self.punc_model_revision = asr_fun_version
        if language == "zh":
            model_asr = os.path.join(asr_root, "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
            self.model_asr = model_asr if os.path.exists(model_asr) else "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            self.model_revision = asr_fun_version
        elif language == "yue":
            model_asr = os.path.join(asr_root, "speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online")
            self.model_asr = model_asr if os.path.exists(model_asr) else "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"
            self.model_revision = "master"
            self.model_vad = self.model_punc = None
            self.vad_model_revision = self.punc_model_revision = None
        else:
            raise ValueError("FunASR does not support this language: " + language)

        self.model = AutoModel(
            model=self.model_asr,
            model_revision=self.model_revision,
            vad_model=self.model_vad,
            vad_model_revision=self.vad_model_revision,
            punc_model=self.model_punc,
            punc_model_revision=self.punc_model_revision,
        )
        self.model_size = model_size
        self.language = language
        self.precision = precision

    def recognize(self, file_list: list, output_file: str) -> EaseVoiceResponse:
        output = []
        trace_data = {}
        for file in tqdm(file_list):
            try:
                text = self.model.generate(input=file)[0]["text"]
                output.append(f"{file}|{self.language.upper()}|{text}")
                trace_data[file] = ResponseStatus.SUCCESS
            except:
                print(traceback.format_exc())
                trace_data[file] = ResponseStatus.FAILED
                return EaseVoiceResponse(status=ResponseStatus.FAILED, message="Failed to recognize audio", data=trace_data)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output))

        return EaseVoiceResponse(status=ResponseStatus.SUCCESS, message="asr success", data=trace_data)


class WhisperAsr(object):
    def __init__(self, model_size: str, language: str, precision: str):
        valid_model_size = [
            "tiny", "tiny.en",
            "base", "base.en",
            "small", "small.en",
            "distil-small.en",
            "medium", "medium.en",
            "distil-medium.en",
            "large-v1",
            "large-v2", "large-v3",
            "large", "distil-large-v2", "distil-large-v3",
            "large-v3-turbo", "turbo"
        ]
        if model_size not in valid_model_size:
            raise ValueError(f"Invalid model size: {model_size}")

        model_path = os.path.join(asr_root, f"faster-whisper-{model_size}")
        self.model_path = model_path if os.path.exists(model_path) else model_size

        valid_language = [
            "af", "am", "ar", "as", "az",
            "ba", "be", "bg", "bn", "bo",
            "br", "bs", "ca", "cs", "cy",
            "da", "de", "el", "en", "es",
            "et", "eu", "fa", "fi", "fo",
            "fr", "gl", "gu", "ha", "haw",
            "he", "hi", "hr", "ht", "hu",
            "hy", "id", "is", "it", "ja",
            "jw", "ka", "kk", "km", "kn",
            "ko", "la", "lb", "ln", "lo",
            "lt", "lv", "mg", "mi", "mk",
            "ml", "mn", "mr", "ms", "mt",
            "my", "ne", "nl", "nn", "no",
            "oc", "pa", "pl", "ps", "pt",
            "ro", "ru", "sa", "sd", "si",
            "sk", "sl", "sn", "so", "sq",
            "sr", "su", "sv", "sw", "ta",
            "te", "tg", "th", "tk", "tl",
            "tr", "tt", "uk", "ur", "uz",
            "vi", "yi", "yo", "zh", "yue",
            "auto"]
        if language not in valid_language:
            raise ValueError(f"Invalid language: {language}")

        self.language = language if language != "auto" else None
        self.precision = precision

    def recognize(self, file_list: list, output_file: str) -> EaseVoiceResponse:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = WhisperModel(self.model_path, device=device, compute_type=self.precision)
        except:
            print(traceback.format_exc())
            return EaseVoiceResponse(status=ResponseStatus.FAILED, message="Failed to load model")

        output = []
        trace_data = {}
        for file in tqdm(file_list):
            try:
                segments, info = model.transcribe(
                    audio=file,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700),
                    language=self.language)
                text = ''.join(str(segment.text) for segment in segments)
                output.append(f"{file}|{info.language.upper()}|{text}")
                trace_data[file] = ResponseStatus.SUCCESS
            except:
                print(traceback.format_exc())
                trace_data[file] = ResponseStatus.FAILED
                return EaseVoiceResponse(status=ResponseStatus.FAILED, message="Failed to recognize audio", data=trace_data)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output))

        return EaseVoiceResponse(status=ResponseStatus.SUCCESS, message="asr success", data=trace_data)
