from dataclasses import dataclass
import logging
import traceback
from typing import Any, List, Tuple
import torch.distributed as dist
import os
from src.train.helper import get_sovits_train_dir, train_ckpt_path, train_logs_path
from src.utils import config
from src.utils import helper
from src.utils.helper import load_json
from src.utils.path import ckpt
from random import randint
from tqdm import tqdm
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch
import warnings
from collections import OrderedDict
from pydantic import BaseModel

from src.easevoice.module import commons
from src.logger import logger
from src.easevoice.module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from src.easevoice.module.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from src.easevoice.module.models import SynthesizerTrn, MultiPeriodDiscriminator
from src.easevoice.module.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler


@dataclass
class SovitsTrainParams:
    batch_size: int = 12
    total_epochs: int = 8
    text_low_lr_rate: float = 0.4
    pretrained_s2G: str = ""
    pretrained_s2D: str = ""
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    save_every_epoch: int = 5
    gpu_ids: str = "0"
    output_model_name: str = ""


class TrainHparams(BaseModel):
    if_save_latest: bool = True
    if_save_every_weights: bool = True
    save_every_epoch: int = 5
    gpu_numbers: str = "0"
    output_dir: str = ""
    train_logs_dir: str = ""
    save_weight_dir: str = ""
    pretrained_s2G: str = ""
    pretrained_s2D: str = ""
    log_interval: int
    eval_interval: int
    seed: int
    epochs: int
    learning_rate: float
    betas: Tuple[float, float]
    eps: float
    batch_size: int
    fp16_run: bool
    lr_decay: float
    segment_size: int
    init_lr_ratio: int
    warmup_epochs: int
    c_mel: int
    c_kl: float
    text_low_lr_rate: float


class DataHparams(BaseModel):
    max_wav_value: float
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    mel_fmin: float
    mel_fmax: Any
    add_blank: bool
    n_speakers: int
    cleaned_text: bool


class ModelHparams(BaseModel):
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    upsample_rates: List[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: List[int]
    n_layers_q: int
    use_spectral_norm: bool
    gin_channels: int
    semantic_frame_rate: str
    freeze_quantizer: bool


class TrainConfig(BaseModel):
    name: str = ""
    train: TrainHparams
    data: DataHparams
    model: ModelHparams
    s2_ckpt_dir: str
    content_module: str


class SovitsTrain:
    def _update_hparams(self, hps: TrainConfig, params: SovitsTrainParams):
        hps.train.batch_size = params.batch_size
        hps.train.epochs = params.total_epochs
        hps.train.text_low_lr_rate = params.text_low_lr_rate

        hps.train.if_save_latest = params.if_save_latest
        hps.train.if_save_every_weights = params.if_save_every_weights
        hps.train.save_every_epoch = params.save_every_epoch
        hps.train.gpu_numbers = params.gpu_ids

        # path
        hps.train.output_dir = get_sovits_train_dir(params.output_model_name)
        hps.train.train_logs_dir = os.path.join(hps.train.output_dir, train_logs_path)
        os.makedirs(hps.train.output_dir, exist_ok=True)
        os.makedirs(hps.train.train_logs_dir, exist_ok=True)
        hps.name = params.output_model_name
        hps.train.save_weight_dir = os.path.join(hps.train.output_dir, train_ckpt_path)
        os.makedirs(hps.train.save_weight_dir, exist_ok=True)

        # set pretrained model path
        if params.pretrained_s2G == "":
            hps.train.pretrained_s2G = config.sovits_pretrained_model_path
        else:
            hps.train.pretrained_s2G = params.pretrained_s2G

        if params.pretrained_s2D == "":
            hps.train.pretrained_s2D = config.sovits_pretrained_model_path.replace("s2G", "s2D")
        else:
            hps.train.pretrained_s2D = params.pretrained_s2D
        return hps

    def __init__(self, params: SovitsTrainParams):
        json_data = load_json(config.s2config_path)
        hps = TrainConfig(**json_data)
        self.hps = self._update_hparams(hps, params)
        self.step = 0
        self.device = "cpu"

        warnings.filterwarnings("ignore")
        os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")  # pyright: ignore
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
        logging.getLogger("numba").setLevel(logging.INFO)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")

    def _save_epoch(self, ckpt: Any, name, epoch, steps, hps: TrainConfig):
        try:
            opt = OrderedDict()
            opt["weight"] = {}
            for key in ckpt.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = ckpt[key].half()
            opt["config"] = hps
            opt["info"] = "%sepoch_%siteration" % (epoch, steps)

            ckpt.save_with_torch(opt, os.path.join(hps.train.save_weight_dir, f"{name}.pth"))
            return "Success"
        except:
            return traceback.format_exc()

    def train(self):
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
        else:
            n_gpus = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(randint(20000, 55555))

        mp.spawn(  # pyright: ignore
            self._run,
            nprocs=n_gpus,
            args=(
                n_gpus,
                self.hps,
            ),
        )

    def _run(self, rank, n_gpus, hps: TrainConfig):
        if rank == 0:
            logger.info(hps)
            writer = SummaryWriter(log_dir=hps.train.train_logs_dir)
            writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.train_logs_dir, "eval"))

        dist.init_process_group(
            backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
            init_method="env://",
            world_size=n_gpus,
            rank=rank,
        )
        torch.manual_seed(hps.train.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        train_dataset = TextAudioSpeakerLoader(hps.data)
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [
                32,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
                1600,
                1700,
                1800,
                1900,
            ],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        collate_fn = TextAudioSpeakerCollate()
        train_loader = DataLoader(
            train_dataset,
            num_workers=6,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            persistent_workers=True,
            prefetch_factor=4,
        )

        net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model.model_dump(),
        ).cuda(rank) if torch.cuda.is_available() else SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model.model_dump(),
        ).to(self.device)

        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank) if torch.cuda.is_available() else MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(self.device)
        for name, param in net_g.named_parameters():
            if not param.requires_grad:
                logger.warning(name, "not requires_grad")

        te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
        et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
        mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
        base_params = filter(
            lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
            net_g.parameters(),
        )

        optim_g = torch.optim.AdamW(
            [
                {"params": base_params, "lr": hps.train.learning_rate},
                {
                    "params": net_g.enc_p.text_embedding.parameters(),
                    "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
                },
                {
                    "params": net_g.enc_p.encoder_text.parameters(),
                    "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
                },
                {
                    "params": net_g.enc_p.mrte.parameters(),
                    "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
                },
            ],
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        if torch.cuda.is_available():
            net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
            net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
        else:
            net_g = net_g.to(self.device)
            net_d = net_d.to(self.device)

        try:
            _, _, _, epoch_str = ckpt.load_checkpoint(
                ckpt.latest_checkpoint_path(hps.train.train_logs_dir, "D_*.pth"),
                net_d,
                optim_d,
            )
            if rank == 0:
                logger.info("loaded D")
            _, _, _, epoch_str = ckpt.load_checkpoint(
                ckpt.latest_checkpoint_path(hps.train.train_logs_dir, "G_*.pth"),
                net_g,
                optim_g,
            )
            global_step = (epoch_str - 1) * len(train_loader)
        except Exception as e:
            logger.warning(f"load failed, exception: {e}, use pretrained instead")
            epoch_str = 1
            global_step = 0
            if hps.train.pretrained_s2G != "" and hps.train.pretrained_s2G != None and os.path.exists(hps.train.pretrained_s2G):
                if rank == 0:
                    logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
                print(
                    net_g.module.load_state_dict(
                        torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                        strict=False,
                    ) if torch.cuda.is_available() else net_g.load_state_dict(
                        torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                        strict=False,
                    )
                )
            if hps.train.pretrained_s2D != "" and hps.train.pretrained_s2D != None and os.path.exists(hps.train.pretrained_s2D):
                if rank == 0:
                    logger.info("loaded pretrained %s" % hps.train.pretrained_s2D)
                print(
                    net_d.module.load_state_dict(
                        torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"]
                    ) if torch.cuda.is_available() else net_d.load_state_dict(
                        torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"]
                    )
                )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=hps.train.lr_decay, last_epoch=-1
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=hps.train.lr_decay, last_epoch=-1
        )
        for _ in range(epoch_str):
            scheduler_g.step()
            scheduler_d.step()

        scaler = GradScaler(enabled=hps.train.fp16_run)

        for epoch in range(epoch_str, hps.train.epochs + 1):
            if rank == 0:
                self._train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d],
                    [optim_g, optim_d],
                    [scheduler_g, scheduler_d],
                    scaler,
                    [train_loader, None],
                    logger,
                    [writer, writer_eval],  # pyright: ignore
                )
            else:
                self._train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d],
                    [optim_g, optim_d],
                    [scheduler_g, scheduler_d],
                    scaler,
                    [train_loader, None],
                    None,
                    None,
                )
            scheduler_g.step()
            scheduler_d.step()

    def _train_and_evaluate(
            self, rank, epoch, hps: TrainConfig, nets, optims, schedulers, scaler, loaders, logger, writers
    ):
        device = self.device
        net_g, net_d = nets
        optim_g, optim_d = optims
        # scheduler_g, scheduler_d = schedulers
        train_loader, eval_loader = loaders
        if writers is not None:
            writer, writer_eval = writers

        train_loader.batch_sampler.set_epoch(epoch)

        net_g.train()
        net_d.train()
        for batch_idx, (
                ssl,
                ssl_lengths,
                spec,
                spec_lengths,
                y,
                y_lengths,
                text,
                text_lengths,
        ) in enumerate(tqdm(train_loader)):
            if torch.cuda.is_available():
                spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
                    rank, non_blocking=True
                )
                y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
                    rank, non_blocking=True
                )
                ssl = ssl.cuda(rank, non_blocking=True)
                ssl.requires_grad = False
                # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
                text, text_lengths = text.cuda(rank, non_blocking=True), text_lengths.cuda(
                    rank, non_blocking=True
                )
            else:
                spec, spec_lengths = spec.to(device), spec_lengths.to(device)
                y, y_lengths = y.to(device), y_lengths.to(device)
                ssl = ssl.to(device)
                ssl.requires_grad = False
                # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
                text, text_lengths = text.to(device), text_lengths.to(device)

            with autocast(enabled=hps.train.fp16_run):
                (
                    y_hat,
                    kl_ssl,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    stats_ssl,
                ) = net_g(ssl, spec, spec_lengths, text, text_lengths)

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if rank == 0:
                if self.step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
                    logger.info(
                        "Train Epoch: {} [{:.0f}%]".format(
                            epoch, 100.0 * batch_idx / len(train_loader)
                        )
                    )
                    logger.info([x.item() for x in losses] + [self.step, lr])

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl_ssl": kl_ssl,
                            "loss/g/kl": loss_kl,
                        }
                    )

                    image_dict = {
                        "slice/mel_org": helper.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": helper.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": helper.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy()
                        ),
                        "all/stats_ssl": helper.plot_spectrogram_to_numpy(
                            stats_ssl[0].data.cpu().numpy()
                        ),
                    }
                    helper.summarize(
                        writer=writer,  # pyright: ignore
                        global_step=self.step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )
            self.step += 1
        if epoch % hps.train.save_every_epoch == 0 and rank == 0:
            if not hps.train.if_save_latest:
                ckpt.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        hps.train.save_weight_dir, f"sovits_G_epoch{epoch}_step{self.step}.pth"
                    ),
                )
                ckpt.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        hps.train.save_weight_dir, f"sovits_D_epoch{epoch}_step{self.step}.pth"
                    ),
                )
            else:
                ckpt.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        hps.train.save_weight_dir, "sovits_G_latest.pth"
                    ),
                )
                ckpt.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(
                        hps.train.save_weight_dir, "sovits_D_latest.pth"
                    ),
                )
            if rank == 0 and hps.train.if_save_every_weights == True:
                if hasattr(net_g, "module"):
                    ckpts = net_g.module.state_dict()
                else:
                    ckpts = net_g.state_dict()
                msg = self._save_epoch(
                    ckpts,
                    hps.name + f"_e{epoch}_s{self.step}",
                    epoch,
                    self.step,
                    hps,
                )
                logger.info(f"saving ckpt {hps.name}_e{epoch}:{msg}")

        if rank == 0:
            logger.info("====> Epoch: {}".format(epoch))
