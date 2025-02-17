#!/usr/bin/env python
# -*- encoding=utf8 -*-
import dataclasses
import traceback

from requests import session

from src.service.session import SessionManager
from src.train.gpt import GPTTrainParams, GPTTrain
from src.train.sovits import SovitsTrain, SovitsTrainParams
from src.utils.response import EaseVoiceResponse, ResponseStatus


def get_update_monitor_data_fn():
    session_manager = SessionManager()
    monitor_data = []

    def update_monitor_data(step: int, data: dict):
        monitor_data.append({"step": step, "data": data})
        session_manager.update_session_info({"train_status": monitor_data})
    return update_monitor_data


class TrainGPTService(object):
    step_name = "train_gpt"

    def __init__(self, gpt_params: GPTTrainParams):
        update_monitor_data = get_update_monitor_data_fn()
        self.gpt_train = GPTTrain(gpt_params, update_monitor_data)

    def train(self) -> EaseVoiceResponse:
        try:
            output = self.gpt_train.train()
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Training GPT completed successfully", data=dataclasses.asdict(output), step_name=self.step_name)
        except Exception as e:
            print(traceback.format_exc(), e)
            return EaseVoiceResponse(ResponseStatus.FAILED, str(e))


class TrainSovitsService(object):
    step_name = "train_sovits"

    def __init__(self, sovits_params: SovitsTrainParams):
        update_monitor_data = get_update_monitor_data_fn()
        self.sovits_train = SovitsTrain(sovits_params, update_monitor_data_fn=update_monitor_data)

    def train(self) -> EaseVoiceResponse:
        try:
            output = self.sovits_train.train()
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Training Sovits completed successfully", data=dataclasses.asdict(output), step_name=self.step_name)
        except Exception as e:
            print(traceback.format_exc(), e)
            return EaseVoiceResponse(ResponseStatus.FAILED, str(e))
