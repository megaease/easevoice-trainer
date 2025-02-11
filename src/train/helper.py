
import datetime
import os
from typing import Optional
from src.utils import config

train_logs_path = "logs"
train_ckpt_path = "ckpt"


def generate_random_name():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_gpt_train_dir(name: Optional[str]):
    if name == "" or name is None:
        name = "gpt_" + generate_random_name()
    return os.path.join(config.all_gpt_train_output_dir, name)


def get_sovits_train_dir(name: Optional[str]):
    if name == "" or name is None:
        name = "sovits_" + generate_random_name()
    return os.path.join(config.all_sovits_train_output_dir, name)


def list_train_gpts():
    return os.listdir(config.all_gpt_train_output_dir)


def list_train_sovits():
    return os.listdir(config.all_sovits_train_output_dir)
