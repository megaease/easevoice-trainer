#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os

import torch
torch.multiprocessing.set_start_method('spawn')
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

            self._init_model_paths()

            self.initialized = True

    def _init_model_paths(self):
        base_path = get_base_path()
        default_dir = os.path.join(base_path, "models", "pretrained")
        logger.info(f"Default pretrained models directory: {default_dir}")
        if not os.path.exists(default_dir):
            logger.warning(f"Default pretrained models directory {default_dir} not exist, please consider to download it before use.")

        self.gpt_path: str = os.environ.get("gpt_path", os.path.join(default_dir, "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"))
        self.bert_path: str = os.environ.get("bert_path", os.path.join(default_dir, "chinese-roberta-wwm-ext-large"))

        self.cnhubert_path: str = os.environ.get("cnhubert_path", os.path.join(default_dir, "chinese-hubert-base"))
        self.sovits_path: str = os.environ.get("sovits_path", os.path.join(default_dir, "gsv-v2final-pretrained", "s2G2333k.pth"))
