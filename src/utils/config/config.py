#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os

import torch
from ..helper import str2bool
from ...logger import logger
from src.utils.path import get_base_path


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

            # is use g2pw for pinyin inference of chinese text
            self.is_g2pw: bool = str2bool(os.environ.get("is_g2pw", "True"))

            base_path = get_base_path()
            default_pretrained_models = os.path.join(base_path, "models", "normalize")
            logger.info(f"Default pretrained models directory: {default_pretrained_models}")
            if not os.path.exists(default_pretrained_models):
                logger.warning(f"Default pretrained models directory {default_pretrained_models} not exist, please consider to download it before use.")
            self.bert_path: str = os.environ.get("bert_path", os.path.join(default_pretrained_models, "chinese-roberta-wwm-ext-large"))

            self.initialized = True
