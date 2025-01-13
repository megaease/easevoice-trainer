#!/usr/bin/env python
# -*- encoding=utf8 -*-

from .config import *
from src.utils.path import get_base_path

base_path = get_base_path()
model_root = "models"
uvr5_root = f"{base_path}/{model_root}/uvr5_weights"
uvr5_params_root = "lib_v5/vr_network/modelparams"
uvr5_onnx_name = "onnx_dereverb_By_FoxJoy"
vocals_output = "vocals"
accompaniments_output = "accompaniments"
slices_output = "slices"
cfg = config.GlobalCFG()

CPU = "cpu"
