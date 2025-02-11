
from dataclasses import dataclass
import datetime
import os
from pathlib import Path
from typing import Optional
from src.utils import config

train_logs_path = "logs"


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
    all_gpts = Path(config.all_gpt_train_output_dir)
    res = {}
    for dir in all_gpts.iterdir():
        if dir.is_dir():
            for file in dir.iterdir():
                if file.is_file() and file.name.endswith(".ckpt"):
                    res[f"{dir.name}/{file.name}"] = str(file)
    return res


def list_train_sovits():
    all_sovits = Path(config.all_sovits_train_output_dir)
    res = {}
    for dir in all_sovits.iterdir():
        if dir.is_dir():
            for file in dir.iterdir():
                if file.is_file() and file.name.endswith(".pth"):
                    res[f"{dir.name}/{file.name}"] = str(file)
    return res


@dataclass
class TrainOutput:
    model_path: str
