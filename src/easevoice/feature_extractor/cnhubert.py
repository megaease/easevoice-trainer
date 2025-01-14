from typing import Optional
import torch.nn as nn
import utils
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
import logging
import time

import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import os
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

logging.getLogger("numba").setLevel(logging.WARNING)


cnhubert_base_path = None


class CNHubert(nn.Module):
    def __init__(self, base_path: Optional[str] = None):
        super().__init__()
        if base_path is None:
            base_path = cnhubert_base_path
        if os.path.exists(base_path):  # pyright: ignore
            ...
        else:
            raise FileNotFoundError(base_path)
        self.model = HubertModel.from_pretrained(base_path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path, local_files_only=True  # pyright: ignore
        )

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]
        return feats


def get_model():
    model = CNHubert()
    model.eval()
    return model


def get_content(hmodel, wav_16k_tensor):
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1, 2)
