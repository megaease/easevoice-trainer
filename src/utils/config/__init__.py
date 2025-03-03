#!/usr/bin/env python
# -*- encoding=utf8 -*-

from .config import *

base_path = get_base_path()
model_root = "models"
uvr5_root = os.path.join(base_path, model_root, "uvr5_weights")
uvr5_params_root = os.path.join("lib_v5", "vr_network", "modelparams")
uvr5_onnx_name = "onnx_dereverb_By_FoxJoy"
vocals_output = "vocals"
accompaniments_output = "accompaniments"
slices_output = "slices"
denoise_root = os.path.join(base_path, model_root, "denoise")
denoises_output = "denoises"
asrs_output = "asrs"
asr_root = os.path.join(base_path, model_root, "asr")
asr_file = "asr.list"
asr_fun_version = "v2.0.4"
refinements_output = "refinements"
refinement_file = "refinement.list"
refinement_root = os.path.join(base_path, model_root, "refinement")
normalize_root = os.path.join(base_path, model_root, "pretrained")
normalize_text = "chinese-roberta-wwm-ext-large"
normalize_ssl = "chinese-hubert-base"
normalize_token = os.path.join("gsv-v2final-pretrained", "s2G2333k.pth")
text_output_name = "2-name2text.txt"
bert_output = "3-bert"
ssl_output = "4-cnhubert"
wav_output = "5-wav32k"
semantic_output = "6-name2semantic.tsv"
s2config_path = os.path.join(base_path, "configs", "s2.json")
gpt_config_path = os.path.join(base_path, "configs", "gpt.yaml")
gpt_pretrained_model_path = os.path.join(normalize_root, "gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
sovits_pretrained_model_path = os.path.join(normalize_root, "gsv-v2final-pretrained", "s2G2333k.pth")
cfg = config.GlobalCFG()
cmd_path = os.path.join(base_path, "src", "cmd")
tb_log_dir = os.path.join(base_path, "tb_logs")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CPU = "cpu"
