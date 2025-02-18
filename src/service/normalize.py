#!/usr/bin/env python
# -*- encoding=utf8 -*-

from dataclasses import dataclass

from src.normalization.normalize import Normalize
from src.utils.response import EaseVoiceResponse, ResponseStatus


@dataclass
class NormalizeParams:
    output_dir: str


class NormalizeService(object):
    def __init__(self, processing_path: str):
        self.processing_path = processing_path

    async def normalize(self) -> EaseVoiceResponse:
        try:
            normalize = Normalize(self.processing_path)
            text_resp = normalize.text()
            if text_resp.status == ResponseStatus.FAILED:
                return text_resp
            ssl_resp = normalize.ssl()
            if ssl_resp.status == ResponseStatus.FAILED:
                return ssl_resp
            token_resp = normalize.token()
            if token_resp.status == ResponseStatus.FAILED:
                return token_resp
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Normalization completed successfully", data={
                "normalize_path": normalize.output_path
            })
        except Exception as e:
            return EaseVoiceResponse(ResponseStatus.FAILED, str(e))
