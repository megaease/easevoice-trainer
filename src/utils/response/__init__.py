#!/usr/bin/env python
# -*- encoding=utf8 -*-


from typing import Literal, Optional

ResponseStatusType = Literal["success", "failed"]

class ResponseStatus:
    """
    Response status code
    """
    SUCCESS = "success"
    FAILED = "failed"


class EaseVoiceResponse(object):
    def __init__(self, status: ResponseStatusType, message: str, data: Optional[dict] = None, uuid: Optional[str] = None):
        self.status = status
        self.message = message
        self.data = data
        self.uuid = uuid

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "data": self.data,
            "uuid": self.uuid
        }

    def __str__(self):
        return str(self.to_dict())
