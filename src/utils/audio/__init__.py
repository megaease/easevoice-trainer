#!/usr/bin/env python
# -*- encoding=utf8 -*-

import os
import traceback
import ffmpeg

import numpy as np

from src.utils.path import format_path


def load_audio(file, sr) -> np.ndarray:
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = format_path(file)
        if not os.path.exists(file):
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        return np.array([])

    return np.frombuffer(out, np.float32).flatten()
