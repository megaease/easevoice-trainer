#!/usr/bin/env python
# -*- encoding=utf8 -*-
from src.service.session import SessionManager, start_train_session_with_spawn
from src.train.sovits import SovitsTrainParams
from src.train.gpt import GPTTrainParams
from src.service.train import do_train_gpt, do_train_sovits
import unittest


class TestTrain(unittest.TestCase):
    session_manager = SessionManager()

    def test_train_gpt(self):
        start_train_session_with_spawn(do_train_gpt, "train_gpt", "TrainGPT", GPTTrainParams(
            train_input_dir="./output",
            output_model_name="test-gpt",
        ))
        print(self.session_manager.get_session_info().get("train_gpt"))

    def test_train_sovits(self):
        start_train_session_with_spawn(do_train_sovits, "train_sovits", "TrainSovits", SovitsTrainParams(
            train_input_dir="./output",
            output_model_name="test-sovits",
        ))
        print(self.session_manager.get_session_info().get("train_sovits"))
