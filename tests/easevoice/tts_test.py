import logging
import multiprocessing
import time
import unittest

from src.easevoice.inference import Runner
from src.logger import logger
from src.service.voice import VoiceCloneService


class TestTTS(unittest.TestCase):
    def setUp(self) -> None:
        logger.setLevel(logging.DEBUG)

    def test_tts_runner(self):
        queue = multiprocessing.Queue()
        Runner(queue)
        queue.put(1)

    def test_voice_clone_service(self):
        voice_service = VoiceCloneService()
        time.sleep(5)
        voice_service.close()


if __name__ == "__main__":
    unittest.main()
