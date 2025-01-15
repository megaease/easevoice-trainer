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
denoise_root = f"{base_path}/{model_root}/denoise"
denoises_output = "denoises"
asrs_output = "asrs"
asr_root = f"{base_path}/{model_root}/asr"
asr_file = "asr.list"
asr_fun_version = "v2.0.4"
cfg = config.GlobalCFG()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CPU = "cpu"
