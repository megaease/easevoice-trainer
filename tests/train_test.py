#!/usr/bin/env python
# -*- encoding=utf8 -*-
from src.train.sovits import SovitsTrainParams
from src.utils.response import ResponseStatus
from src.train.gpt import GPTTrainParams
from src.service.train import TrainGPTService, TrainSovitsService
import unittest


class TestTrain(unittest.TestCase):
    gpt_service = TrainGPTService(gpt_params=GPTTrainParams(
        processing_path="./output",
        normalize_path="./output/test",
        output_model_name="test",
    ))
    sovits_service = TrainSovitsService(sovits_params=SovitsTrainParams(
        normalize_path="./output/test",
        output_model_name="test",
    ))

    def test_train(self):
        resp = self.gpt_service.train()
        self.assertEqual(resp.status, ResponseStatus.SUCCESS)
