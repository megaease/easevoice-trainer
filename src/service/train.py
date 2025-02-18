#!/usr/bin/env python
# -*- encoding=utf8 -*-
import dataclasses
from multiprocessing import Queue
import traceback

from requests import session

from src.service.session import SessionManager
from src.train.gpt import GPTTrainParams, GPTTrain
from src.train.sovits import SovitsTrain, SovitsTrainParams
from src.utils.response import EaseVoiceResponse, ResponseStatus


async def do_train_gpt(params: GPTTrainParams):
    service = TrainGPTService(params)
    return service.train()


class TrainGPTService(object):
    def __init__(self, gpt_params: GPTTrainParams):
        self.gpt_train = GPTTrain(gpt_params)

    def train(self) -> EaseVoiceResponse:
        try:
            output = self.gpt_train.train()
            resp = EaseVoiceResponse(ResponseStatus.SUCCESS, "Training GPT completed successfully", data=dataclasses.asdict(output))
        except Exception as e:
            print(traceback.format_exc(), e)
            resp = EaseVoiceResponse(ResponseStatus.FAILED, str(e))
        return resp


async def do_train_sovits(params: SovitsTrainParams):
    service = TrainSovitsService(params)
    return service.train()


class TrainSovitsService(object):
    def __init__(self, sovits_params: SovitsTrainParams):
        self.sovits_train = SovitsTrain(sovits_params)

    def train(self) -> EaseVoiceResponse:
        try:
            output = self.sovits_train.train()
            resp = EaseVoiceResponse(ResponseStatus.SUCCESS, "Training Sovits completed successfully", data=dataclasses.asdict(output))
        except Exception as e:
            print(traceback.format_exc(), e)
            resp = EaseVoiceResponse(ResponseStatus.FAILED, str(e))
        return resp
