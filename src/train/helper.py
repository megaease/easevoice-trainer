
from dataclasses import dataclass
import datetime
import os
from pathlib import Path
from typing import Optional

from src.logger import logger

train_logs_path = "logs"


def generate_random_name():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _get_all_gpt_train_output_dir(project_dir: str):
    return os.path.join(project_dir, "models", "gpt_train")


def get_gpt_train_dir(project_dir: str, name: Optional[str]):
    if name == "" or name is None:
        name = "gpt_" + generate_random_name()
    return os.path.join(_get_all_gpt_train_output_dir(project_dir), name)


def _get_all_sovits_train_output_dir(project_dir: str):
    return os.path.join(project_dir, "models", "sovits_train")


def get_sovits_train_dir(project_dir: str, name: Optional[str]):
    if name == "" or name is None:
        name = "sovits_" + generate_random_name()
    return os.path.join(_get_all_sovits_train_output_dir(project_dir), name)


def list_train_gpts(project_dir: str):
    try:
        all_gpts = Path(_get_all_gpt_train_output_dir(project_dir))
        res = {}
        for dir in all_gpts.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
                    if file.is_file() and file.name.endswith(".ckpt"):
                        res[f"{dir.name}/{file.name}"] = str(file)
        return res
    except Exception as e:
        logger.warning(f"list_train_gpts failed: {e}")
        return {}


def list_train_sovits(project_dir: str):
    try:
        all_sovits = Path(_get_all_sovits_train_output_dir(project_dir))
        res = {}
        for dir in all_sovits.iterdir():
            if dir.is_dir():
                for file in dir.iterdir():
                    if file.is_file() and file.name.endswith(".pth"):
                        res[f"{dir.name}/{file.name}"] = str(file)
        return res
    except Exception as e:
        logger.warning(f"list_train_sovits failed: {e}")
        return {}


@dataclass
class TrainOutput:
    model_path: str
