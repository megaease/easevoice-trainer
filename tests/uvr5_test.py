#!/usr/bin/env python
# -*- encoding=utf8 -*-

import unittest


class TestUvr5(unittest.TestCase):
    def test_uvr5_service(self):
        from src.service.audio import AudioService
        from src.utils.response import ResponseStatus

        model_name = "onnx_dereverb_By_FoxJoy"
        input_dir = "./resources"
        output_dir = "./output"
        audio_format = "wav"
        response = AudioService.uvr5(model_name, input_dir, output_dir, audio_format)
        self.assertEqual(response.status, ResponseStatus.SUCCESS)
