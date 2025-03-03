#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os

import torch
from huggingface_hub import snapshot_download

from src.utils.path import get_base_path
from ..helper import str2bool
from ...logger import logger


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
        pretrained_dir = os.path.join(base_path, "models", "pretrained")
        logger.info(f"Default pretrained models directory: {pretrained_dir}")
        if not os.path.exists(pretrained_dir):
            snapshot_download("lj1995/GPT-SoVITS", resume_download=True, local_dir=pretrained_dir)
        asr_dir = os.path.join(base_path, "models", "uvr5_weights")
        logger.info(f"Default asr models directory: {asr_dir}")
        if not os.path.exists(asr_dir):
            snapshot_download("Delik/uvr5_weights", resume_download=True, local_dir=asr_dir)

        self.gpt_path: str = os.environ.get("gpt_path", os.path.join(pretrained_dir, "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"))
        self.bert_path: str = os.environ.get("bert_path", os.path.join(pretrained_dir, "chinese-roberta-wwm-ext-large"))

        self.cnhubert_path: str = os.environ.get("cnhubert_path", os.path.join(pretrained_dir, "chinese-hubert-base"))
        self.sovits_path: str = os.environ.get("sovits_path", os.path.join(pretrained_dir, "gsv-v2final-pretrained", "s2G2333k.pth"))
