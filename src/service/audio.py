#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os
import traceback
from dataclasses import dataclass

import ffmpeg
import numpy as np
import torch
from scipy.io import wavfile

from src.audiokit.asr import FunAsr, WhisperAsr
from src.audiokit.denoise import Denoise
from src.audiokit.refinement import Refinement
from src.audiokit.slicer import Slicer
from src.audiokit.uvr5.separate import SeparateBase, SeparateMDXNet, SeparateMDXC, SeparateVR, SeparateVREcho
from src.utils.audio import load_audio
from src.utils.config import vocals_output, slices_output, denoises_output, asrs_output, asr_file, refinements_output, refinement_file, accompaniments_output
from src.utils.response import ResponseStatus, EaseVoiceResponse


@dataclass
class AudioUVR5Params:
    source_dir: str
    output_dir: str
    model_name: str
    audio_format: str


@dataclass
class AudioSlicerParams:
    source_dir: str
    output_dir: str
    threshold: int = -34
    min_length: int = 4000
    min_interval: int = 300
    hop_size: int = 10
    max_silent_kept: int = 500
    normalize_max: float = 0.9
    alpha_mix: float = 0.25


@dataclass
class AudioDenoiseParams:
    source_dir: str
    output_dir: str


@dataclass
class AudioASRParams:
    source_dir: str
    output_dir: str
    asr_model: str = "funasr"
    model_size: str = "large"
    language: str = "zh"
    precision: str = "float32"


@dataclass
class AudioRefinementSubmitParams:
    source_dir: str
    output_dir: str
    source_file_path: str
    language: str
    text_content: str


@dataclass
class AudioRefinementDeleteParams:
    source_dir: str
    output_dir: str
    source_file_path: str

@dataclass
class AudioRefinementReloadParams:
    source_dir: str
    output_dir: str


class AudioService(object):
    def __init__(self, source_dir: str, output_dir: str):
        super().__init__()
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.refinement = Refinement(os.path.join(self.output_dir, asrs_output, asr_file), os.path.join(self.output_dir, refinements_output, refinement_file))

    def uvr5(self, model_name: str = "HP5_only_main_vocal", audio_format: str = "wav", **kwargs) -> EaseVoiceResponse:
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
                    print(traceback.format_exc())
                if need_reformat == 1:
                    tmp_path = "%s/%s.reformatted.wav" % (self.source_dir, file_name.split(".")[0])
                    os.system(f'ffmpeg -i "{input_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
                try:
                    if done == 0:
                        separator.separate("%s.reformatted.wav" % file_name.split(".")[0])
                        trace_data[file_name] = ResponseStatus.SUCCESS
                except:
                    print(traceback.format_exc())
                    trace_data[file_name] = ResponseStatus.FAILED
        except:
            return EaseVoiceResponse(ResponseStatus.FAILED, traceback.format_exc(), trace_data)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "UVR5 Success", trace_data)

    def slicer(
            self,
            threshold: int = -34,
            min_length: int = 4000,
            min_interval: int = 300,
            hop_size: int = 10,
            max_silent_kept: int = 500,
            normalize_max: float = 0.9,
            alpha_mix: float = 0.25,
    ) -> EaseVoiceResponse:
        os.makedirs(os.path.join(self.output_dir, slices_output), exist_ok=True)
        files = self._get_files(vocals_output)
        files.extend(self._get_files(accompaniments_output))
        slicer = Slicer(
            sr=32000,
            threshold=int(threshold),
            min_length=int(min_length),
            min_interval=int(min_interval),
            hop_size=int(hop_size),
            max_sil_kept=int(max_silent_kept),
        )
        data = {}
        for file in files:
            name = os.path.basename(file)
            name = name.split(".")[0]
            try:
                audio = load_audio(file, 32000)
                if audio.shape[0] == 0:
                    continue
                for chunk, start, end in slicer.slice(audio):
                    nor_max = np.abs(chunk).max()
                    if nor_max > 1:
                        chunk /= nor_max
                    chunk = (chunk / nor_max * (normalize_max * alpha_mix)) + (1 - alpha_mix) * chunk
                    output_filename = "%s_%010d_%010d.wav" % (name, start, end)
                    output_path = os.path.join(self.output_dir, slices_output, output_filename)
                    wavfile.write(output_path, 32000, (chunk * 32767).astype(np.int16))
                data[name] = ResponseStatus.SUCCESS
            except:
                print(file, " slice failed ", traceback.format_exc())
                EaseVoiceResponse(ResponseStatus.FAILED, "Slice Failed", {"file_name": name})
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Slice Success", data)

    def denoise(self) -> EaseVoiceResponse:
        os.makedirs(os.path.join(self.output_dir, denoises_output), exist_ok=True)
        trace_data = {}
        try:
            files = self._get_files(slices_output)

            denoise = Denoise()
            for file_name in files:
                base_name = os.path.basename(file_name)
                output_path = os.path.join(self.output_dir, denoises_output, base_name)
                denoise.denoise(file_name, output_path)
                trace_data[file_name] = ResponseStatus.SUCCESS
        except:
            print(traceback.format_exc())
            return EaseVoiceResponse(ResponseStatus.FAILED, traceback.format_exc(), trace_data)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Denoise Success", trace_data)

    def asr(self, asr_model: str = "funasr", model_size: str = "large", language: str = "zh", precision: str = "float32") -> EaseVoiceResponse:
        file_list = self._get_files(denoises_output)
        output_file = os.path.join(self.output_dir, asrs_output, asr_file)
        dump_file = os.path.join(self.output_dir, refinements_output, refinement_file)
        os.makedirs(os.path.join(self.output_dir, asrs_output), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, refinements_output), exist_ok=True)
        if asr_model == "faster-whisper":
            model = WhisperAsr(model_size, language, precision)
            return model.recognize(file_list, output_file, dump_file)
        elif asr_model == "funasr":
            model = FunAsr(model_size, language, precision)
            return model.recognize(file_list, output_file, dump_file)
        else:
            return EaseVoiceResponse(ResponseStatus.FAILED, "ASR model not supported", {})

    def _get_files(self, output_path: str):
        files = []
        for name in sorted(list(os.listdir(os.path.join(self.output_dir, output_path)))):
            file_path = os.path.join(self.output_dir, output_path, name)
            if os.path.isfile(file_path) and file_path.split(".")[-1] in ["wav", "flac", "mp3", "m4a"]:
                files.append(file_path)
        return files

    def refinement_reload_source(self) -> EaseVoiceResponse:
        try:
            resp = self.refinement.reload_text()
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Reload Source Success", resp)
        except Exception as e:
            return EaseVoiceResponse(ResponseStatus.FAILED, "Reload Source Failed", {
                "error": str(e)
            })

    def refinement_load_source(self) -> EaseVoiceResponse:
        os.makedirs(os.path.join(self.output_dir, refinements_output), exist_ok=True)
        if len(self.refinement.source_file_content) == 0:
            self.refinement.load_text()
        resp = self.refinement.source_file_content
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Load Source Success", resp)

    def refinement_submit_text(self, index: str, language: str, text_content: str) -> EaseVoiceResponse:
        self.refinement.submit_text(index, language.lower(), text_content)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Submit Text Success", self.refinement.source_file_content)

    def refinement_delete_text(self, file_index: str):
        self.refinement.delete_text(file_index)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Delete Text Success", self.refinement.source_file_content)
