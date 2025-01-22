#!/usr/bin/env python
# -*- encoding=utf8 -*-

from src.normalization.normalize import Normalize
from src.utils.response import EaseVoiceResponse, ResponseStatus


class NormalizeService(object):
    def __init__(self, processing_path: str):
        self.processing_path = processing_path

    def normalize(self) -> EaseVoiceResponse:
        try:
            normalize = Normalize(self.processing_path)
            return normalize.text()
        except Exception as e:
            return EaseVoiceResponse(ResponseStatus.FAILED, str(e))
