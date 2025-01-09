#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os
import traceback
from operator import truediv

import ffmpeg
import torch

from src.audiokit.uvr5.separate import SeparateBase, SeparateMDXNet, SeparateMDXC, SeparateVR, SeparateVREcho
from src.utils.response import ResponseStatus, EaseVoiceResponse


class AudioService(object):
    @staticmethod
    def uvr5(model_name: str, input_dir: str, output_dir: str, audio_format: str, **kwargs) -> EaseVoiceResponse:
        trace_data = {}
        try:
            base_separator = SeparateBase(
                model_name=model_name,
                input_dir=input_dir,
                output_dir=output_dir,
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

            files = [name for name in os.listdir(input_dir)]
            for file_name in files:
                input_path = os.path.join(input_dir, file_name)
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
                    tmp_path = "%s/%s.reformatted.wav" % (input_dir, file_name.split(".")[0])
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
