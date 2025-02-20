#!/usr/bin/env python
# -*- encoding=utf8 -*-
import dataclasses
import traceback
from src.train.gpt import GPTTrainParams, GPTTrain
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
            resp = EaseVoiceResponse(ResponseStatus.SUCCESS, "Training GPT completed successfully", data=dataclasses.asdict(output)) # pyright: ignore
        except Exception as e:
            print(traceback.format_exc(), e)
            resp = EaseVoiceResponse(ResponseStatus.FAILED, str(e)) # pyright: ignore
        return resp
