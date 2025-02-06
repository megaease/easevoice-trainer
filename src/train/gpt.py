#!/usr/bin/env python
# -*- encoding=utf8 -*-

import logging
import os
import platform
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from src.easevoice.soundstorm.auto_reg.data.data_module import Text2SemanticDataModule
from src.easevoice.soundstorm.auto_reg.models.t2s_lightning_module import Text2SemanticLightningModule
from src.utils.config import gpt_config_path, train_output, cfg, gpt_pretrained_model_path, semantic_output, text_output_name, train_gpt_logs_output


@dataclass
class GPTTrainParams:
    batch_size: int = 12
    total_epochs: int = 15
    save_every_epoch: int = 5
    if_dpo: bool = False
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    gpu_ids: str = "0"
    model_path: str = gpt_pretrained_model_path
    processing_path: str = ""
    normalize_path: str = ""
    output_model_name: str = "gpt"


class GPTCheckpoint(ModelCheckpoint):
    def __init__(
            self,
            config,
            if_save_latest,
            if_save_every_weights,
            half_weights_save_dir,
            output_name,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.output_name = output_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if (
                    self._every_n_epochs >= 1
                    and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            ):
                to_clean = []
                if (
                        self.if_save_latest == True
                ):
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    state_dict = trainer.strategy._lightning_module.state_dict()
                    for key in state_dict:
                        to_save_od["weight"][key] = state_dict[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        new_path = os.path.join(
                            self.half_weights_save_dir,
                            "%s-e%s.ckpt" % (self.output_name, trainer.current_epoch + 1),
                        )
                        torch.save(to_save_od, new_path)
            self._save_last_checkpoint(trainer, monitor_candidates)


class GPTTrain(object):
    def __init__(self, params: GPTTrainParams):
        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        torch.set_float32_matmul_precision("high")
        with open(gpt_config_path, "r") as f:
            data = f.read()
            self.config = yaml.load(data, Loader=yaml.FullLoader)
        self.processing_path = params.processing_path
        self.normalize_path = params.normalize_path
        self.train_output = os.path.join(params.normalize_path, train_output)
        self.train_logs_output = os.path.join(params.normalize_path, train_gpt_logs_output)
        self.train_ckpts_output = os.path.join(self.train_logs_output, "ckpt")
        os.makedirs(self.train_output, exist_ok=True)
        os.makedirs(self.train_logs_output, exist_ok=True)
        os.makedirs(self.train_ckpts_output, exist_ok=True)
        self.cfg = cfg
        if not self.cfg.is_half:
            self.config["train"]["precision"] = "32"
            params.batch_size = max(1, params.batch_size // 2)
        self.config["train"]["batch_size"] = params.batch_size
        self.config["train"]["epochs"] = params.total_epochs
        self.config["train"]["save_every_n_epoch"] = params.save_every_epoch
        self.config["train"]["if_dpo"] = params.if_dpo
        self.config["train"]["if_save_latest"] = params.if_save_latest
        self.config["train"]["if_save_every_weights"] = params.if_save_every_weights
        self.config["pretrained_s1"] = params.model_path
        self.config["train"]["half_weights_save_dir"] = self.train_output
        self.config["train_semantic_path"] = os.path.join(params.normalize_path, semantic_output)
        self.config["train_phoneme_path"] = os.path.join(params.normalize_path, text_output_name)
        self.config["logs_output_dir"] = self.train_logs_output
        self.config["train"]["output_name"] = params.output_model_name
        os.environ["hz"] = "25hz"
        seed_everything(self.config["train"]["seed"], workers=True)
        ckpt_callback: ModelCheckpoint = GPTCheckpoint(
            config=self.config,
            if_save_latest=self.config["train"]["if_save_latest"],
            if_save_every_weights=self.config["train"]["if_save_every_weights"],
            half_weights_save_dir=self.config["train"]["half_weights_save_dir"],
            output_name=self.config["train"]["output_name"],
            save_top_k=-1,
            monitor="top_3_acc",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=self.config["train"]["save_every_n_epoch"],
            dirpath=self.train_ckpts_output,
        )
        logger = TensorBoardLogger(name="log", save_dir=self.train_logs_output)
        os.environ["MASTER_ADDR"] = "localhost"
        self.trainer: Trainer = Trainer(
            max_epochs=self.config["train"]["epochs"],
            accelerator=self.cfg.device,
            limit_val_batches=0,
            devices=-1 if torch.cuda.is_available() else 1,
            benchmark=False,
            fast_dev_run=False,
            strategy=DDPStrategy(
                process_group_backend="nccl" if platform.system() != "Windows" else "gloo"
            ) if torch.cuda.is_available() else "auto",
            precision=self.config["train"]["precision"],
            logger=logger,
            num_sanity_val_steps=0,
            callbacks=[ckpt_callback],
            use_distributed_sampler=False,
        )
        self.model: Text2SemanticLightningModule = Text2SemanticLightningModule(
            self.config, Path(self.train_logs_output)
        )

        self.data_module: Text2SemanticDataModule = Text2SemanticDataModule(
            self.config,
            train_semantic_path=self.config["train_semantic_path"],
            train_phoneme_path=self.config["train_phoneme_path"],
        )
        trainer_ckpt_path = self._get_newest_ckpt(os.listdir(self.train_ckpts_output))
        self.trainer_ckpt_path = os.path.join(self.train_ckpts_output, trainer_ckpt_path) if trainer_ckpt_path else None

    def train(self):
        self.trainer.fit(self.model, self.data_module, ckpt_path=self.trainer_ckpt_path)

    @staticmethod
    def _get_newest_ckpt(file_list: []):
        if file_list is None or len(file_list) == 0:
            return None

        pattern = r'epoch=(\d+)-step=(\d+)\.ckpt'
        extracted_info = []
        for string in file_list:
            match = re.match(pattern, string)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                extracted_info.append((epoch, step, string))
        sorted_info = sorted(
            extracted_info, key=lambda x: (x[0], x[1]), reverse=True)
        newest_ckpt = sorted_info[0][2]
        return newest_ckpt
