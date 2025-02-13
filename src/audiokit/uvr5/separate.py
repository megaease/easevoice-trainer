#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os

import torch
import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torch.nn as nn
import onnxruntime as ort

from src.utils.config import uvr5_root, cfg, uvr5_params_root, CPU, uvr5_onnx_name, vocals_output, accompaniments_output
from src.utils.path import format_path, get_parent_abs_path
from src.audiokit.uvr5.lib_v5.vr_network.model_param_init import ModelParameters
import src.audiokit.uvr5.lib_v5.vr_network.nets as Nets
from src.utils.response import ResponseStatus, EaseVoiceResponse
import src.audiokit.uvr5.lib_v5.vr_network.spec_utils as spec_utils
from src.audiokit.uvr5.lib_v5.vr_network.nets_new import CascadedNet
from src.audiokit.uvr5.lib_v5.vr_network.bs_roformer import BSRoformer
from src.audiokit.uvr5.lib_v5.vr_network.mdxnet import ConvTDFNetTrim


class SeparateBase:
    def __init__(self, model_name: str, input_dir: str, output_dir: str, audio_format: str, **kwargs):
        self.model_name = model_name
        self.model_path = os.path.join(uvr5_root, f"{model_name}.pth")
        self.cfg = cfg
        self.input_dir = format_path(input_dir)
        self.output_dir = format_path(output_dir)
        self.output_vocal_dir = os.path.join(self.output_dir, vocals_output)
        self.output_accompaniment_dir = os.path.join(self.output_dir, accompaniments_output)
        self.audio_format = audio_format
        self.kwargs = kwargs
        self.file_list = []
        if self.input_dir is not None:
            self.file_list = [f for f in os.listdir(self.input_dir) if f.endswith(audio_format)]
        if self.output_dir is not None:
            os.makedirs(self.output_vocal_dir, exist_ok=True)
            os.makedirs(self.output_accompaniment_dir, exist_ok=True)
        self.accompaniment_head = "accompaniments_"
        self.vocal_head = "vocals_"
        self.reverse_output = kwargs.get("reverse_output", False)

    def separate(self, file_name: str) -> EaseVoiceResponse:
        pass

    def write_output(self, data: np.ndarray, sr: int, name: str, is_vocal: bool, extend_path: str = None):
        name = name.split(".")[0]
        path = "{}_{}".format(name, extend_path) if extend_path else name
        is_vocal = is_vocal if not self.reverse_output else not is_vocal
        head = self.vocal_head if is_vocal else self.accompaniment_head
        if self.audio_format in ["wav", "flac"]:
            sf.write(
                os.path.join(
                    self.output_vocal_dir if is_vocal else self.output_accompaniment_dir,
                    f"{head}{path}.{self.audio_format}",
                ),
                data,
                sr,
            )
        else:
            file_path = os.path.join(
                self.output_vocal_dir if is_vocal else self.output_accompaniment_dir,
                f"{head}{path}.wav",
            )
            sf.write(path, data, sr)
            opt_format_path = file_path[:-4] + f".{self.audio_format}"
            if os.path.exists(path):
                os.system(f"ffmpeg -i {path} -vn {opt_format_path} -q:a 2 -y")
                if os.path.exists(opt_format_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass


class SeparateVR(SeparateBase):
    step_name = "uvr5"
    def __init__(self, base_instance: SeparateBase, **kwargs):
        super().__init__(base_instance.model_name, base_instance.input_dir, base_instance.output_dir, base_instance.audio_format, **base_instance.kwargs, **kwargs)
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

        model = Nets.get_nets_model(self.mp.param['bins'] * 2)
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
            return EaseVoiceResponse(ResponseStatus.FAILED, "Input or output directory is not provided", step_name=self.step_name)
        x_wave, y_wave, x_spec_s, y_spec_s = {}, {}, {}, {}
        band_n = len(self.mp.param["band"])
        input_high_end = None
        input_high_end_head = None
        for index in range(band_n, 0, -1):
            band_params = self.mp.param["band"][index]
            if index == band_n:
                (x_wave[index], _,) = librosa.core.load(
                    os.path.join(self.input_dir, file_name),
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
            x_spec_s[index] = spec_utils.wave_to_spectrogram_mt(
                x_wave[index],
                band_params["hl"],
                band_params["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
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
        self.write_output(
            extend_path="{}".format(self.data["agg"]),
            data=(np.array(wav_accompaniment) * 32768).astype("int16"),
            sr=self.mp.param["sr"],
            name=file_name,
            is_vocal=False,
        )

        # separate vocal
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], v_spec_m, input_high_end, self.mp
            )
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                v_spec_m, self.mp, input_high_end_head, input_high_end_
            )
        else:
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)

        self.write_output(
            extend_path="{}".format(self.data["agg"]),
            data=(np.array(wav_vocals) * 32768).astype("int16"),
            sr=self.mp.param["sr"],
            name=file_name,
            is_vocal=True,
        )
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Separation completed", step_name=self.step_name)

    def inference(self, x_spec, device, model, aggressiveness, data):
        def _execute(_x_mag_pad, _roi_size, _n_window, _device, _model, _aggressiveness, _is_half=True):
            _model.eval()
            with torch.no_grad():
                predicts = []

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
            _x_mag = np.abs(_x_spec)
            _x_phase = np.angle(_x_spec)

            return _x_mag, _x_phase

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


class SeparateVREcho(SeparateVR):
    def __init__(self, base_instance: SeparateBase, **kwargs):
        SeparateBase.__init__(self, base_instance.model_name, base_instance.input_dir, base_instance.output_dir, base_instance.audio_format, **base_instance.kwargs, **kwargs)
        self._parent_directory = get_parent_abs_path(__file__)
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": kwargs.get("tta", False),
            # Constants
            "window_size": 512,
            "agg": kwargs.get("agg", 10),
            "high_end_process": "mirroring",
        }
        self.mp = ModelParameters("{}/{}/4band_v3.json".format(self._parent_directory, uvr5_params_root))
        nout = 64 if "DeReverb" in self.model_path else 48
        model = CascadedNet(self.mp.param["bins"] * 2, nout)
        cpk = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(cpk)
        model.eval()
        if self.cfg.is_half:
            model.half().to(self.cfg.device)
        else:
            model.to(self.cfg.device)
        self.model = model


class SeparateMDXNet(SeparateBase):
    step_name = "uvr5"
    def __init__(self, base_instance: SeparateBase, **kwargs):
        super().__init__(base_instance.model_name, base_instance.input_dir, base_instance.output_dir, base_instance.audio_format, **base_instance.kwargs, **kwargs)
        self.onnx = f"{uvr5_root}/{uvr5_onnx_name}"
        self.shifts = 10  # 'Predict with randomised equivariant stabilisation'
        self.mixing = "min_mag"  # ['default','min_mag','max_mag']
        self.chunks = 15
        self.margin = 44100
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.model_ = ConvTDFNetTrim(
            device=self.cfg.device,
            model_name="Conv-TDF",
            target_name="vocals",
            L=11,
            dim_f=self.dim_f,
            dim_t=self.dim_t,
            n_fft=self.n_fft,
        )
        self.model = ort.InferenceSession(
            os.path.join(self.onnx, self.model_.target_name + ".onnx"),
            providers=[
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.margin
        chunk_size = self.chunks * 44100
        assert not margin == 0, "margin cannot be zero!"
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1

            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)

            start = skip - s_margin

            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        """
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        """
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")
        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i: i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(torch.device("cpu"))
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.denoise:
                    spec_pred = (
                            -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                            + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0])
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                if margin_size == 0:
                    end = None
                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        # del self.model
        progress_bar.close()
        return _sources

    def separate(self, file_name: str) -> EaseVoiceResponse:
        mix, rate = librosa.load(os.path.join(self.input_dir, file_name), mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T
        self.write_output(
            data=mix - opt,
            sr=rate,
            name=file_name,
            is_vocal=True,
        )
        self.write_output(
            data=opt,
            sr=rate,
            name=file_name,
            is_vocal=False,
        )
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Separation completed", step_name=self.step_name)


class SeparateMDXC(SeparateBase):
    step_name = "uvr5"
    def __init__(self, base_instance: SeparateBase, **kwargs):
        super().__init__(base_instance.model_name, base_instance.input_dir, base_instance.output_dir, base_instance.audio_format, **base_instance.kwargs, **kwargs)
        model = self.get_model_from_config()
        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        if self.cfg.is_half:
            self.model = model.half().to(self.cfg.device)
        else:
            self.model = model.to(self.cfg.device)

    @staticmethod
    def get_model_from_config():
        config = {
            "attn_dropout": 0.1,
            "depth": 12,
            "dim": 512,
            "dim_freqs_in": 1025,
            "dim_head": 64,
            "ff_dropout": 0.1,
            "flash_attn": True,
            "freq_transformer_depth": 1,
            "freqs_per_bands": (
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12,
                12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 128, 129),
            "heads": 8,
            "linear_transformer_depth": 0,
            "mask_estimator_depth": 2,
            "multi_stft_hop_size": 147,
            "multi_stft_normalized": False,
            "multi_stft_resolution_loss_weight": 1.0,
            "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
            "num_stems": 1,
            "stereo": True,
            "stft_hop_length": 441,
            "stft_n_fft": 2048,
            "stft_normalized": False,
            "stft_win_length": 2048,
            "time_transformer_depth": 1,

        }

        model = BSRoformer(
            **dict(config)
        )

        return model

    def demix_track(self, model, mix, device):
        c = 352800
        # num_overlap
        n = 1
        fade_size = c // 10
        step = int(c // n)
        border = c - step
        batch_size = 4

        length_init = mix.shape[-1]

        progress_bar = tqdm(total=length_init // step + 1)
        progress_bar.set_description("Processing")

        # Do pad from the beginning and end to account floating window results better
        if length_init > 2 * border and (border > 0):
            mix = nn.functional.pad(mix, (border, border), mode='reflect')

        # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
        window_size = c
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window_start = torch.ones(window_size)
        window_middle = torch.ones(window_size)
        window_finish = torch.ones(window_size)
        window_start[-fade_size:] *= fadeout  # First audio chunk, no fadein
        window_finish[:fade_size] *= fadein  # Last audio chunk, no fadeout
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein

        with torch.amp.autocast('cuda'):
            with torch.inference_mode():
                req_shape = (1,) + tuple(mix.shape)

                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)
                i = 0
                batch_data = []
                batch_locations = []
                while i < mix.shape[1]:
                    part = mix[:, i:i + c].to(device)
                    length = part.shape[-1]
                    if length < c:
                        if length > c // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, c - length), mode='reflect')
                        else:
                            part = nn.functional.pad(input=part, pad=(0, c - length, 0, 0), mode='constant', value=0)
                    if self.cfg.is_half:
                        part = part.half()
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step
                    progress_bar.update(1)

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0)
                        x = model(arr)

                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish

                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start:start + l] += x[j][..., :l].cpu() * window[..., :l]
                            counter[..., start:start + l] += window[..., :l]

                        batch_data = []
                        batch_locations = []

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                if length_init > 2 * border and (border > 0):
                    # Remove pad
                    estimated_sources = estimated_sources[..., border:-border]

        progress_bar.close()

        return {k: v for k, v in zip(['vocals', 'other'], estimated_sources)}

    def separate(self, file_name: str) -> EaseVoiceResponse:
        self.model.eval()
        path = os.path.join(self.input_dir, file_name)

        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
        except Exception:
            return EaseVoiceResponse(ResponseStatus.FAILED, "Failed to load audio file", step_name=self.step_name)

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        mix_orig = mix.copy()

        mixture = torch.tensor(mix, dtype=torch.float32)
        res = self.demix_track(self.model, mixture, self.cfg.device)

        estimates = res['vocals'].T
        self.write_output(
            data=estimates,
            sr=sr,
            name=file_name,
            is_vocal=True,
        )
        self.write_output(
            data=mix_orig.T - estimates,
            sr=sr,
            name=file_name,
            is_vocal=False,
        )
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Separation completed", step_name=self.step_name)
