import torch.nn as nn
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
import logging
import os
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()

logging.getLogger("numba").setLevel(logging.WARNING)


class CNHubert(nn.Module):
    def __init__(self, base_path: str, eval: bool = False):
        super().__init__()
        if not os.path.exists(base_path):
            raise FileNotFoundError(base_path)

        self.model = HubertModel.from_pretrained(base_path, local_files_only=True)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            base_path, local_files_only=True
        )
        if eval:
            self.eval()

    def forward(self, x):
        input_values = self.feature_extractor(
            x, return_tensors="pt", sampling_rate=16000
        ).input_values.to(x.device)
        feats = self.model(input_values)["last_hidden_state"]  # pyright: ignore
        return feats
