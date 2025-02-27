# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
from inflect import ten
from pytorch_lightning import LightningModule
import torch
from typing import Any, Dict, Optional
import os
import sys

from src.utils.helper import convert_tensor_to_python
from src.utils.helper.connector import MultiProcessOutputConnector

from ..modules.optim import ScaledAdam
from ..modules.lr_schedulers import WarmupCosineLRSchedule
from ..models.t2s_model import Text2SemanticDecoder

sys.path.append(os.getcwd())


class Text2SemanticLightningModule(LightningModule):
    def __init__(self, config, output_dir, is_train=True):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
        pretrained_s1 = config.get("pretrained_s1")
        if pretrained_s1 and is_train:
            print("train gpt pretrained_s1 with",
                  self.load_state_dict(
                      torch.load(pretrained_s1, map_location="cpu")["weight"]
                  )
                  )
        if is_train:
            self.automatic_optimization = False
            self.save_hyperparameters()
            self.eval_dir = output_dir / "eval"
            self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.connector = MultiProcessOutputConnector()
        self._connector_step = 0

    def training_step(self, batch: Dict, batch_idx: int):  # pyright: ignore
        opt: Any = self.optimizers()
        scheduler: Any = self.lr_schedulers()
        forward = self.model.forward if self.config["train"].get("if_dpo", False) == True else self.model.forward_old
        loss, acc = forward(
            batch["phoneme_ids"],
            batch["phoneme_ids_len"],
            batch["semantic_ids"],
            batch["semantic_ids_len"],
            batch["bert_feature"],
        )
        self.manual_backward(loss)
        if batch_idx > 0 and batch_idx % 4 == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()

        self.log(
            "total_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "lr",
            scheduler.get_last_lr()[0],
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"top_{self.top_k}_acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        print(f"current global rank {self.global_rank}, current local rank {self.local_rank}, current global step {self.global_step}, current epoch {self.current_epoch}")
        if self.global_rank == 0:
            self.connector.write_loss(
                step=self._connector_step,
                loss=convert_tensor_to_python(loss),
                other={"acc": convert_tensor_to_python(acc), "lr": convert_tensor_to_python(scheduler.get_last_lr()[0]), "epoch": self.current_epoch, }
            )
            self._connector_step += 1

    def validation_step(self, batch: Dict, batch_idx: int):
        return

    def configure_optimizers(self):
        model_parameters = self.model.parameters()
        parameters_names = []
        parameters_names.append(
            [name_param_pair[0] for name_param_pair in self.model.named_parameters()]
        )
        lm_opt = ScaledAdam(
            model_parameters,
            lr=0.01,
            betas=(0.9, 0.95),
            clipping_scale=2.0,
            parameters_names=parameters_names,
            show_dominant_parameters=False,
            clipping_update_period=1000,
        )

        return {
            "optimizer": lm_opt,
            "lr_scheduler": {
                "scheduler": WarmupCosineLRSchedule(
                    lm_opt,
                    init_lr=self.config["optimizer"]["lr_init"],
                    peak_lr=self.config["optimizer"]["lr"],
                    end_lr=self.config["optimizer"]["lr_end"],
                    warmup_steps=self.config["optimizer"]["warmup_steps"],
                    total_steps=self.config["optimizer"]["decay_steps"],
                )
            },
        }
