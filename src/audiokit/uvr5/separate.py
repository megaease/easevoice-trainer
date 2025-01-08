#!/usr/bin/env python
# -*- encoding=utf8 -*-
import sys

sys.path.append("..")

from src.utils.config import uvr5_root, cfg
from src.utils.path import format_path


class SeparateAttributes:
    def __init__(self, model_name: str, input_dir: str, output_dir: str, audio_format: str, **kwargs):
        self.model_name = model_name
        self.model_path = f"{uvr5_root}/{model_name}.pth"
        self.cfg = cfg
        self.input_dir = format_path(input_dir)
        self.output_dir = format_path(output_dir)
        self.output_vocal_dir = f"{self.output_dir}/vocals"
        self.output_accompaniment_dir = f"{self.output_dir}/accompaniment"
        self.audio_format = audio_format
        self.kwargs = kwargs


class SeparateVR(SeparateAttributes):
    pass


class SeparateMDXNet(SeparateAttributes):
    pass


class SeparateMDXC(SeparateAttributes):
    pass
