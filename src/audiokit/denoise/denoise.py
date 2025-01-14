#!/usr/bin/env python
# -*- encoding=utf8 -*-

import os

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from src.utils.config import denoises_output, denoise_root


class Denoise(object):
    def __init__(self):
        model_dir = "speech_frcrn_ans_cirm_16k"
        self.model_path = os.path.join(denoise_root, model_dir)
        if not os.path.exists(self.model_path):
            self.model_path = "damo/speech_frcrn_ans_cirm_16k"
        self.pipeline = pipeline(Tasks.acoustic_noise_suppression, model=self.model_path)

    def denoise(self, source_file: str, output_file: str):
        self.pipeline(source_file, output_path=output_file)
