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
    def __init__(self, status: ResponseStatus, message: str, data: Optional[dict] = None, step_name: Optional[str] = None):
        self.status = status
        self.message = message
        self.data = data
        self.step_name = step_name

    def to_dict(self):
        return {
            "status": self.status.value,
            "message": self.message,
            "data": self.data,
            "step_name": self.step_name
        }

    def __str__(self):
        return str(self.to_dict())
