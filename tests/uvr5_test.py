#!/usr/bin/env python
# -*- encoding=utf8 -*-

import unittest

from src.service.audio import AudioService


class TestUvr5(unittest.TestCase):
    service = AudioService(source_dir="./resources", output_dir="./output")
    def test_uvr5_service(self):
        from src.service.audio import AudioService
        from src.utils.response import ResponseStatus

        model_name = "HP5-主旋律人声vocals+其他instrumentals"
        audio_format = "wav"
        response = self.service.uvr5(model_name, audio_format)
        self.assertEqual(response.status, ResponseStatus.SUCCESS)
