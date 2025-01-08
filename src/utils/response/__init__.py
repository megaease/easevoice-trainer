#!/usr/bin/env python
# -*- encoding=utf8 -*-


# Response status code
class ResponseStatus(object):
    SUCCESS = "success"
    FAILED = "failed"


class EaseVoiceResponse(object):
    def __init__(self, status: ResponseStatus, message: str, data: dict = None):
        self.status = status
        self.message = message
        self.data = data

    def to_dict(self):
        return {
            "status": self.status,
            "message": self.message,
            "data": self.data
        }

    def __str__(self):
        return str(self.to_dict())
