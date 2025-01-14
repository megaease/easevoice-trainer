#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os.path
import traceback

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

from src.utils.config import asr_root
from src.utils.response import EaseVoiceResponse, ResponseStatus


class FunAsr(object):
    def __init__(self):
        pass

    def recognize(self, file_list: list) -> EaseVoiceResponse:
        pass


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
        for file in file_list:
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
