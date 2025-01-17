#!/usr/bin/env python
# -*- encoding=utf8 -*-
import multiprocessing
import os
import traceback
from multiprocessing import Process

import ffmpeg
import torch
import numpy as np
from joblib.externals.loky.backend.queues import Queue
from scipy.io import wavfile

from src.audiokit.uvr5.separate import SeparateBase, SeparateMDXNet, SeparateMDXC, SeparateVR, SeparateVREcho
from src.utils.response import ResponseStatus, EaseVoiceResponse
from src.audiokit.slicer import Slicer
from src.audiokit.denoise import Denoise
from src.audiokit.asr import FunAsr, WhisperAsr
from src.audiokit.refinement import Refinement, Labeling
from src.utils.audio import load_audio
from src.utils.config import vocals_output, slices_output, denoises_output, asrs_output, asr_file, refinements_output, refinement_file
from src.service.task import TaskService
from src.api.api import Task, TaskStatus, AudioServiceSteps


class AudioService(TaskService):
    def __init__(self, source_dir: str, output_dir: str, task: Task):
        super().__init__()
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.refinement = Refinement(os.path.join(self.output_dir, asrs_output, asr_file), os.path.join(self.output_dir, refinements_output, refinement_file))
        self.task = task

    def uvr5(self, model_name: str, audio_format: str, progress=None, **kwargs) -> EaseVoiceResponse:
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
            total = len(files)
            for file_name in files:
                if progress is not None and callable(progress):
                    progress(int((files.index(file_name) + 1) * 100 / total))
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
                        separator.separate("%s.reformatted.wav" % file_name.split(".")[0])
                except:
                    traceback.print_exc()
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
            num_process: int = 4,
    ) -> EaseVoiceResponse:
        os.makedirs(os.path.join(self.output_dir, slices_output), exist_ok=True)
        files = self._get_files(vocals_output)
        process = []
        queue = multiprocessing.Queue()
        for i in range(num_process):
            file_list = files[i::num_process]
            if len(file_list) == 0:
                continue
            p = Process(target=self.slice_audio, args=(threshold, min_length, min_interval, hop_size, max_silent_kept, normalize_max, alpha_mix, file_list, queue))
            p.start()
            process.append(p)
        for p in process:
            p.join()

        results = {}
        all_success = True
        while not queue.empty():
            resp = queue.get()
            results.update(resp.data)
            if resp.status == ResponseStatus.FAILED:
                all_success = False

        if not all_success:
            return EaseVoiceResponse(ResponseStatus.FAILED, "Slice Failed", results)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Slice Success", results)

    def slice_audio(self, threshold: int, min_length: int, min_interval: int, hop_size: int, max_silent_kept: int, normalize_max: float, alpha_mix: float, file_list: list, queue: Queue):
        slicer = Slicer(
            sr=32000,
            threshold=int(threshold),
            min_length=int(min_length),
            min_interval=int(min_interval),
            hop_size=int(hop_size),
            max_sil_kept=int(max_silent_kept),
        )
        for file in file_list:
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
                queue.put(EaseVoiceResponse(ResponseStatus.SUCCESS, "Success", {"file_name": name}))
            except:
                print(file, " slice failed ", traceback.format_exc())
                queue.put(EaseVoiceResponse(ResponseStatus.FAILED, traceback.format_exc(), {"file_name": name}))
        return

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
        os.makedirs(os.path.join(self.output_dir, asrs_output), exist_ok=True)
        if asr_model == "faster-whisper":
            model = WhisperAsr(model_size, language, precision)
            return model.recognize(file_list, output_file)
        elif asr_model == "funasr":
            model = FunAsr(model_size, language, precision)
            return model.recognize(file_list, output_file)
        else:
            return EaseVoiceResponse(ResponseStatus.FAILED, "ASR model not supported", {})

    def _get_files(self, output_path: str):
        files = []
        for name in sorted(list(os.listdir(os.path.join(self.output_dir, output_path)))):
            file_path = os.path.join(self.output_dir, output_path, name)
            if os.path.isfile(file_path) and file_path.split(".")[-1] in ["wav", "flac", "mp3", "m4a"]:
                files.append(file_path)
        return files

    def refinement_load_source(self) -> EaseVoiceResponse:
        os.makedirs(os.path.join(self.output_dir, refinements_output), exist_ok=True)
        if len(self.refinement.source_file_content) == 0:
            self.refinement.load_text()
            self.refinement.save_file()
        resp = self.refinement.source_file_content
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Load Source Success", resp)

    def refinement_submit_text(self, index: str, language: str, text_content: str) -> EaseVoiceResponse:
        self.refinement.submit_text(index, language.upper(), text_content)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Submit Text Success", self.refinement.source_file_content)

    def refinement_delete_text(self, file_index: str):
        self.refinement.delete_text(file_index)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Delete Text Success", self.refinement.source_file_content)

    def refinement_reload(self):
        self.refinement.load_text()
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Reload Success", self.refinement.source_file_content)

    def audio_service(self):
        uvr_resp = self.uvr5(self.task.args["model_name"], self.task.args["audio_format"], self._progress)
        if not self._handle_resp(uvr_resp, AudioServiceSteps.Slicer):
            return

        slice_resp = self.slicer(
            threshold=self.task.args["threshold"] if "threshold" in self.task.args else -34,
            min_length=self.task.args["min_length"] if "min_length" in self.task.args else 4000,
            min_interval=self.task.args["min_interval"] if "min_interval" in self.task.args else 300,
            hop_size=self.task.args["hop_size"] if "hop_size" in self.task.args else 10,
            max_silent_kept=self.task.args["max_silent_kept"] if "max_silent_kept" in self.task.args else 500,
            normalize_max=self.task.args["normalize_max"] if "normalize_max" in self.task.args else 0.9,
            alpha_mix=self.task.args["alpha_mix"] if "alpha_mix" in self.task.args else 0.25,
            num_process=self.task.args["num_process"] if "num_process" in self.task.args else 4,
        )
        if not self._handle_resp(slice_resp, AudioServiceSteps.Denoise):
            return

        denoise_resp = self.denoise()
        if not self._handle_resp(denoise_resp, AudioServiceSteps.ASR):
            return

        asr_resp = self.asr(
            asr_model=self.task.args["asr_model"] if "asr_model" in self.task.args else "funasr",
            model_size=self.task.args["model_size"] if "model_size" in self.task.args else "large",
            language=self.task.args["language"] if "language" in self.task.args else "zh",
            precision=self.task.args["precision"] if "precision" in self.task.args else "float32",
        )
        if not self._handle_resp(asr_resp, ""):
            return

    def _progress(self, progress: int):
        self.task.progress.current_step_progress = progress
        self.submit_task(self.task)
        return

    def _handle_resp(self, response: EaseVoiceResponse, next_step: str) -> bool:
        if response.status == ResponseStatus.FAILED:
            self.task.progress.status = TaskStatus.FAILED
            self.submit_task(self.task)
            return False
        self.task.progress.completed_steps += 1
        if next_step == "":
            self.task.progress.status = TaskStatus.COMPLETED
        else:
            self.task.progress.current_step = next_step
        self.submit_task(self.task)
        return True
