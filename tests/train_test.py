#!/usr/bin/env python
# -*- encoding=utf8 -*-

import unittest

from src.service.train import TrainGPTService
from src.train.gpt import GPTTrainParams
from src.utils.response import ResponseStatus


class TestTrain(unittest.TestCase):
    service = TrainGPTService(gpt_params=GPTTrainParams(
        processing_path="./output",
        normalize_path="./output/test",
        output_model_name="test",
    ))

    def test_train(self):
        resp = self.service.train()
        self.assertEqual(resp.status, ResponseStatus.SUCCESS)
