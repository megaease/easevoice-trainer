#!/usr/bin/env python
# -*- encoding=utf8 -*-

import unittest

from src.service.normalize import NormalizeService
from src.utils.response import ResponseStatus


class TestNormalize(unittest.TestCase):
    service = NormalizeService(processing_path="./output")

    def test_normalize(self):
        response = self.service.normalize()
        self.assertEqual(response.status, ResponseStatus.SUCCESS)
