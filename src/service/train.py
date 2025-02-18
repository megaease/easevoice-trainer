#!/usr/bin/env python
# -*- encoding=utf8 -*-
import dataclasses
from multiprocessing import Queue
import traceback

from requests import session

from src.service.session import SessionManager
from src.train.gpt import GPTTrainParams, GPTTrain
from src.train.helper import TrainMonitorQueue
from src.train.sovits import SovitsTrain, SovitsTrainParams
from src.utils.response import EaseVoiceResponse, ResponseStatus


def do_train_gpt(params: GPTTrainParams, train_monitor_queue: Queue):
    service = TrainGPTService(params, train_monitor_queue)
    return service.train()


class TrainGPTService(object):
    step_name = "train_gpt"

    def __init__(self, gpt_params: GPTTrainParams, queue: Queue):
        self.train_monitor_queue = TrainMonitorQueue(queue)
        self.gpt_train = GPTTrain(gpt_params, self.train_monitor_queue)

    def train(self) -> EaseVoiceResponse:
        try:
            output = self.gpt_train.train()
            resp = EaseVoiceResponse(ResponseStatus.SUCCESS, "Training GPT completed successfully", data=dataclasses.asdict(output), step_name=self.step_name)
        except Exception as e:
            print(traceback.format_exc(), e)
            resp = EaseVoiceResponse(ResponseStatus.FAILED, str(e))
        finally:
            self.train_monitor_queue.close()
        return resp


def do_train_sovits(params: SovitsTrainParams, queue: Queue):
    service = TrainSovitsService(params, queue)
    return service.train()


class TrainSovitsService(object):
    step_name = "train_sovits"

    def __init__(self, sovits_params: SovitsTrainParams, queue: Queue):
        self.train_monitor_queue = TrainMonitorQueue(queue)
        self.sovits_train = SovitsTrain(sovits_params, self.train_monitor_queue)

    def train(self) -> EaseVoiceResponse:
        try:
            output = self.sovits_train.train()
            resp = EaseVoiceResponse(ResponseStatus.SUCCESS, "Training Sovits completed successfully", data=dataclasses.asdict(output), step_name=self.step_name)
        except Exception as e:
            print(traceback.format_exc(), e)
            resp = EaseVoiceResponse(ResponseStatus.FAILED, str(e))
        finally:
            self.train_monitor_queue.close()
        return resp
