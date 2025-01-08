#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os
import sys

import torch
import librosa
import numpy as np
from tqdm import tqdm

sys.path.append("..")

from src.utils.config import uvr5_root, cfg, uvr5_params_root, CPU
from src.utils.path import format_path, get_parent_abs_path
from src.audiokit.uvr5.lib_v5.vr_network.model_param_init import ModelParameters
import src.audiokit.uvr5.lib_v5.vr_network.nets as nets
from src.utils.response import ResponseStatus, EaseVoiceResponse
import src.audiokit.uvr5.lib_v5.vr_network.spec_utils as spec_utils


class SeparateAttributes:
    def __init__(self, model_name: str, input_dir: str, output_dir: str, audio_format: str, **kwargs):
        self.model_name = model_name
        self.model_path = f"{uvr5_root}/{model_name}.pth"
        self.cfg = cfg
        self.input_dir = format_path(input_dir)
        self.output_dir = format_path(output_dir)
        self.output_vocal_dir = f"{self.output_dir}/vocals"
        self.output_accompaniment_dir = f"{self.output_dir}/accompaniment"
        self.audio_format = audio_format
        self.kwargs = kwargs
        self.file_list = []
        if self.input_dir is not None:
            self.file_list = [f for f in os.listdir(self.input_dir) if f.endswith(audio_format)]
        if self.output_dir is not None:
            os.makedirs(self.output_vocal_dir, exist_ok=True)
            os.makedirs(self.output_accompaniment_dir, exist_ok=True)


class SeparateVR(SeparateAttributes):
    def __init__(self, model_name: str, input_dir: str, output_dir: str, audio_format: str, **kwargs):
        super().__init__(model_name, input_dir, output_dir, audio_format, **kwargs)
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": kwargs.get("tta", False),
            # Constants
            "window_size": 512,
            "agg": kwargs.get("agg", 10),
            "high_end_process": "mirroring",
        }
        self._parent_directory = get_parent_abs_path(__file__)
        self.mp = ModelParameters("{}/{}/4band_v2.json".format(self._parent_directory, uvr5_params_root))

        model = nets.CascadedASPPNet(self.mp.param["bins"] * 2)
        cpk = torch.load(self.model_path, map_location=CPU)
        model.load_state_dict(cpk)
        model.eval()
        if self.cfg.is_half:
            model.half().to(self.cfg.device)
        else:
            model.to(self.cfg.device)
        self.model = model

    def separate(self, file_name: str) -> EaseVoiceResponse:
        if self.input_dir is None or self.output_dir is None:
            return EaseVoiceResponse(ResponseStatus.FAILED, "Input or output directory is not provided")
        base_name = os.path.basename(file_name)
        x_wave, y_wave, x_spec_s, y_spec_s = {}, {}, {}, {}
        band_n = len(self.mp.param["band"])
        input_high_end = None
        input_high_end_head = None
        for index in range(band_n, 0, -1):
            band_params = self.mp.param["band"][index]
            if index == band_n:
                (x_wave[index], _,) = librosa.core.load(
                    file_name,
                    sr=band_params["sr"],
                    mono=False,
                    dtype=np.float32,
                    res_type=band_params["res_type"],
                )
                if x_wave[index].ndim == 1:
                    x_wave[index] = np.asfortranarray([x_wave[index], x_wave[index]])
            else:
                x_wave[index] = librosa.core.resample(
                    x_wave[index + 1],
                    orig_sr=self.mp.param["band"][index + 1]["sr"],
                    target_sr=band_params["sr"],
                    res_type=band_params["res_type"],
                )
            x_spec_s[index] = spec_utils.wave_to_spectrogram(
                x_wave[index],
                band_params["hl"],
                band_params["n_fft"],
                self.mp,
                index,
            )
            if index == band_n and self.data["high_end_process"] != "none":
                input_high_end_head = (band_params["n_fft"] // 2 - band_params["crop_stop"]) + (
                        self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = x_spec_s[index][:, band_params["n_fft"] // 2 - input_high_end_head: band_params["n_fft"] // 2, :]

        x_spec_m = spec_utils.combine_spectrograms(x_spec_s, self.mp)
        aggressive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggressive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
            'aggr_correction': self.mp.param.get('aggr_correction'),
        }
        with torch.no_grad():
            pred, x_mag, x_phase = self.inference(
                x_spec_m, self.cfg.device, self.model, aggressiveness, self.data
            )
        if self.data["postprocess"]:
            pred_inv = np.clip(x_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * x_phase
        v_spec_m = x_spec_m - y_spec_m

        # separate accompaniment
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], y_spec_m, input_high_end, self.mp
            )
            wav_accompaniment = spec_utils.cmb_spectrogram_to_wave(
                y_spec_m, self.mp, input_high_end_head, input_high_end_
            )
        else:
            wav_accompaniment = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)

    def inference(self, x_spec, device, model, aggressiveness, data):
        def _execute(_x_mag_pad, _roi_size, _n_window, _device, _model, _aggressiveness, _is_half=True):
            _model.eval()
            with torch.no_grad():
                predicts = []

                iterations = [_n_window]

                total_iterations = sum(iterations)
                for i in tqdm(range(_n_window)):
                    start = i * _roi_size
                    x_mag_window = _x_mag_pad[
                                   None, :, :, start: start + data["window_size"]
                                   ]
                    x_mag_window = torch.from_numpy(x_mag_window)
                    if _is_half:
                        x_mag_window = x_mag_window.half()
                    x_mag_window = x_mag_window.to(_device)

                    predict = _model.predict(x_mag_window, _aggressiveness)

                    predict = predict.detach().cpu().numpy()
                    predicts.append(predict[0])

                predict = np.concatenate(predicts, axis=2)
            return predict

        def preprocess(_x_spec):
            x_mag = np.abs(_x_spec)
            x_phase = np.angle(_x_spec)

            return x_mag, x_phase

        x_mag, x_phase = preprocess(x_spec)

        coef = x_mag.max()
        x_mag_pre = x_mag / coef

        n_frame = x_mag_pre.shape[2]
        pad_l, pad_r, roi_size = self.make_padding(n_frame, data["window_size"], model.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        x_mag_pad = np.pad(x_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        if list(model.state_dict().values())[0].dtype == torch.float16:
            is_half = True
        else:
            is_half = False
        pred = _execute(
            x_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
        )
        pred = pred[:, :, :n_frame]

        if data["tta"]:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1

            x_mag_pad = np.pad(x_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

            pred_tta = _execute(
                x_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
            )
            pred_tta = pred_tta[:, :, roi_size // 2:]
            pred_tta = pred_tta[:, :, :n_frame]

            return (pred + pred_tta) * 0.5 * coef, x_mag, np.exp(1.0j * x_phase)
        else:
            return pred * coef, x_mag, np.exp(1.0j * x_phase)

    @staticmethod
    def make_padding(width, crop_size, offset):
        left = offset
        roi_size = crop_size - left * 2
        if roi_size == 0:
            roi_size = crop_size
        right = roi_size - (width % roi_size) + left

        return left, right, roi_size


class SeparateMDXNet(SeparateAttributes):
    pass


class SeparateMDXC(SeparateAttributes):
    pass
