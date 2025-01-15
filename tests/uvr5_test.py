#!/usr/bin/env python
# -*- encoding=utf8 -*-

import unittest

from src.service.audio import AudioService
from src.utils.response import ResponseStatus


class TestUvr5(unittest.TestCase):
    service = AudioService(source_dir="./resources", output_dir="./output")

    def test_uvr5_service(self):
        model_name = "HP5_only_main_vocal"
        audio_format = "wav"
        response = self.service.uvr5(model_name, audio_format)
        self.assertEqual(response.status, ResponseStatus.SUCCESS)

    def test_slicer_service(self):
        response = self.service.slicer()
        self.assertEqual(response.status, ResponseStatus.SUCCESS)

    def test_denoise_service(self):
        response = self.service.denoise()
        self.assertEqual(response.status, ResponseStatus.SUCCESS)

    def test_asr_service(self):
        response = self.service.asr()
        self.assertEqual(response.status, ResponseStatus.SUCCESS)

    def test_refinement_service(self):
        self.service.refinement_load_source()
        self.service.refinement_reload()
