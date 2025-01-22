import logging
import multiprocessing
import unittest

from src.easevoice.inference import Runner
from src.logger import logger


class TestTTS(unittest.TestCase):
    def setUp(self) -> None:
        logger.setLevel(logging.DEBUG)

    def test_tts_runner(self):
        queue = multiprocessing.Queue()
        Runner(queue)
        queue.put(1)


if __name__ == "__main__":
    unittest.main()
