#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os

import torch


class GlobalCFG(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalCFG, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.is_half: bool = os.environ.get("is_half", "True").lower() == 'true'
            self.is_share: bool = os.environ.get("is_share", "False").lower() == 'true'
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                if (
                        ("16" in gpu_name and "V100" not in gpu_name.upper())
                        or "P40" in gpu_name.upper()
                        or "P10" in gpu_name.upper()
                        or "1060" in gpu_name
                        or "1070" in gpu_name
                        or "1080" in gpu_name
                ):
                    self.is_half = False
            if self.device == "cpu":
                self.is_half = False

            self.initialized = True
