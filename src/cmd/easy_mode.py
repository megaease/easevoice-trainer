#!/usr/bin/env python
# -*- encoding=utf8 -*-
import sys
sys.path.append('.')
sys.path.append('..')

from src.rest.types import TaskCMD
from src.utils.response import EaseVoiceResponse, ResponseStatus
from src.utils.helper.connector import ConnectorDataType, MultiProcessOutputConnector
from src.train.gpt import GPTTrainParams, GPTTrain
from dataclasses import asdict
import traceback
import json
import argparse
from src.train.sovits import SovitsTrain, SovitsTrainParams
from src.service.normalize import NormalizeService
from src.api.api import EaseVoiceRequest
from src.service.audio import AudioService
from src.utils import config
import os
import subprocess
import tempfile
from typing import Any



def _check_response(connector: MultiProcessOutputConnector, response: EaseVoiceResponse, step_name: str, current_step: int):
    connector.write_session_data({
        "current_step": current_step,
    })
    if response.status == ResponseStatus.FAILED:
        connector.write_session_data({
            "current_step_description": f"{step_name} failed: {response.message}",
        })
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.FAILED, message=f"{step_name} failed: {response.message}"))
        sys.exit(1)

    connector.write_session_data({
        "current_step_description": f"{step_name} completed successfully",
        "progress": current_step / 7 * 100,
    })


def _run_train(cmd_file: str, request: Any):
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", delete=False) as fp:
        params = asdict(request)
        params = json.dumps(params)
        fp.write(params)
        temp_file_path = fp.name

    proc = subprocess.Popen(
        [sys.executable, os.path.join(config.cmd_path, cmd_file), "-c", temp_file_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=config.base_path,
    )
    connector = MultiProcessOutputConnector()
    for data in connector.read_data(proc):
        if data.dataType == ConnectorDataType.RESP:
            return data.response
    return EaseVoiceResponse(ResponseStatus.FAILED, "Unknown error")


def main():
    connector = MultiProcessOutputConnector()
    try:
        parser = argparse.ArgumentParser(
            description="run easy mode",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=argparse.FileType("r"),
            required=True,
        )
        args = parser.parse_args()
        config_content = args.config.read()
        args.config.close()
        config = json.loads(config_content)

        params = EaseVoiceRequest(**config)
        session_data = {
            "total_steps": 7,
            "current_step": 0,
            "progress": 0,
            "current_step_description": "Prepare for starting EaseVoice",
        }
        connector.write_session_data(session_data)
        output_dir = os.path.join(params.source_dir, "easy_mode")
        os.makedirs(output_dir, exist_ok=True)

        audio_service = AudioService(source_dir=params.source_dir, output_dir=str(output_dir))
        resp = audio_service.uvr5()
        _check_response(connector, resp, "Audio UVR5", 1)
        resp = audio_service.slicer()
        _check_response(connector, resp, "Audio Slicer", 2)
        resp = audio_service.denoise()
        _check_response(connector, resp, "Audio Denoise", 3)
        resp = audio_service.asr()
        _check_response(connector, resp, "Audio ASR", 4)
        normalize_service = NormalizeService(processing_path=output_dir)
        resp = normalize_service.normalize()
        _check_response(connector, resp, "Normalization", 5)
        normalize_path = resp.data["normalize_path"]  # pyright: ignore

        sovits_name = params.sovits_output_name
        sovits_params = SovitsTrainParams(train_input_dir=normalize_path, output_model_name=sovits_name)
        sovits_resp = _run_train(TaskCMD.tran_sovits, sovits_params)
        print(f"sovits resp of easy mode: {sovits_resp}")
        _check_response(connector, resp, "Sovits Training", 6)

        gpt_name = params.gpt_output_name
        gpt_params = GPTTrainParams(train_input_dir=normalize_path, output_model_name=gpt_name)
        gpt_resp = _run_train(TaskCMD.train_gpt, gpt_params)
        print(f"gpt resp of easy mode: {gpt_resp}")
        _check_response(connector, resp, "GPT Training", 7)
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.SUCCESS, message="FTraining GPT completed successfully", data={
            "sovits_output": sovits_resp.data["model_path"], # pyright: ignore
            "gpt_output": gpt_resp.data["model_path"], # pyright: ignore
        }))
    except Exception as e:
        print(traceback.format_exc(), e)
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.FAILED, message=f"{e}"))


if __name__ == "__main__":
    main()
