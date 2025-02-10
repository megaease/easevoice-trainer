#!/usr/bin/env python
# -*- encoding=utf8 -*-
import traceback

from src.train.gpt import GPTTrainParams, GPTTrain
from src.train.sovits import SovitsTrain, SovitsTrainParams
from src.utils.response import EaseVoiceResponse, ResponseStatus


class TrainGPTService(object):
    def __init__(self, gpt_params: GPTTrainParams):
        self.gpt_train = GPTTrain(gpt_params)

    def train(self) -> EaseVoiceResponse:
        try:
            self.gpt_train.train()
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Training GPT completed successfully", data={
                "model_path": self.gpt_train.train_output
            })
        except Exception as e:
            print(traceback.format_exc(), e)
            return EaseVoiceResponse(ResponseStatus.FAILED, str(e))


class TrainSovitsService(object):
    def __init__(self, sovits_params: SovitsTrainParams):
        self.sovits_train = SovitsTrain(sovits_params)

    def train(self) -> EaseVoiceResponse:
        try:
            self.sovits_train.train()
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Training Sovits completed successfully")
        except Exception as e:
            print(traceback.format_exc(), e)
            return EaseVoiceResponse(ResponseStatus.FAILED, str(e))
