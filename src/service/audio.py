#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os
import traceback
from multiprocessing import Process

import ffmpeg
import torch
import numpy as np
from scipy.io import wavfile

from src.audiokit.uvr5.separate import SeparateBase, SeparateMDXNet, SeparateMDXC, SeparateVR, SeparateVREcho
from src.utils.response import ResponseStatus, EaseVoiceResponse
from src.audiokit.slicer import Slicer
from src.utils.audio import load_audio
from src.utils.config import vocals_output, slices_output


class AudioService(object):
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = source_dir
        self.output_dir = output_dir

    def uvr5(self, model_name: str, audio_format: str, **kwargs) -> EaseVoiceResponse:
        trace_data = {}
        try:
            base_separator = SeparateBase(
                model_name=model_name,
                input_dir=self.source_dir,
                output_dir=self.output_dir,
                audio_format=audio_format,
                reverse_output="HP3" in model_name,
                **kwargs
            )
            if model_name == "onnx_dereverb_By_FoxJoy":
                separator = SeparateMDXNet(base_separator)
            elif model_name == "Bs_Roformer" or "bs_roformer" in model_name.lower():
                separator = SeparateMDXC(base_separator)
            else:
                if "DeEcho" in model_name:
                    separator = SeparateVREcho(base_separator)
                else:
                    separator = SeparateVR(base_separator)

            files = [name for name in os.listdir(self.source_dir)]
            for file_name in files:
                input_path = os.path.join(self.source_dir, file_name)
                if not os.path.isfile(input_path):
                    continue
                need_reformat = 1
                done = 0
                try:
                    info = ffmpeg.probe(input_path, cmd="ffprobe")
                    if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":
                        need_reformat = 0
                        separator.separate(file_name)
                        done = 1
                        trace_data[file_name] = ResponseStatus.SUCCESS
                except:
                    need_reformat = 1
                    traceback.print_exc()
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (self.source_dir, file_name.split(".")[0])
                    os.system(f'ffmpeg -i "{input_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
                try:
                    if done == 0:
                        separator.separate("%s.reformatted.wav" % file_name)
                except:
                    traceback.print_exc()
                    trace_data[file_name] = ResponseStatus.FAILED
        except:
            return EaseVoiceResponse(ResponseStatus.FAILED, traceback.format_exc(), trace_data)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Success", trace_data)

    def slicer(self, threshold: int, min_length: int, min_interval: int, hop_size: int, max_silent_kept: int, normalize_max: float, alpha_mix: float, num_process: int):
        os.makedirs(os.path.join(self.output_dir, slices_output), exist_ok=True)
        files = [os.path.join(self.output_dir, vocals_output, name) for name in sorted(list(os.listdir(os.path.join(self.output_dir, vocals_output))))]
        process = []
        for i in range(num_process):
            file_list = files[i::num_process]
            p = Process(target=self.slice_audio, args=(threshold, min_length, min_interval, hop_size, max_silent_kept, normalize_max, alpha_mix, file_list))
            p.start()
            process.append(p)
        for p in process:
            p.join()

    def slice_audio(self, threshold: int, min_length: int, min_interval: int, hop_size: int, max_silent_kept: int, normalize_max: float, alpha_mix: float, file_list: list):
        slicer = Slicer(
            sr=32000,
            threshold=int(threshold),
            min_length=int(min_length),
            min_interval=int(min_interval),
            hop_size=int(hop_size),
            max_sil_kept=int(max_silent_kept),
        )
        for file in file_list:
            try:
                name = os.path.basename(file)
                audio = load_audio(file, 32000)
                if audio.shape[0] == 0:
                    continue
                for chunk, start, end in slicer.slice(audio):
                    nor_max = np.abs(chunk).max()
                    if nor_max > 1:
                        chunk /= nor_max
                    chunk = (chunk / nor_max * (normalize_max * alpha_mix)) + (1 - alpha_mix) * chunk
                    output_path = os.path.join(self.output_dir, slices_output, "%s_%010d_%010d.wav" % (name, start, end))
                    wavfile.write(output_path, 32000, (chunk * 32767).astype(np.int16))
            except:
                print(file, " slice failed ", traceback.format_exc())
        return
