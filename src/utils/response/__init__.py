#!/usr/bin/env python
# -*- encoding=utf8 -*-


from enum import Enum
from typing import Optional


class ResponseStatus(Enum):
    """
    Response status code
    """
    SUCCESS = "success"
    FAILED = "failed"


class EaseVoiceResponse(object):
    def __init__(self, status: ResponseStatus, message: str, data: Optional[dict] = None):
        self.status = status
        self.message = message
        self.data = data

    def to_dict(self):
        return {
            "status": self.status.value,
            "message": self.message,
            "data": self.data
        }

    def __str__(self):
        return str(self.to_dict())
