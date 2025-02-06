from concurrent.futures import thread
import gc
import multiprocessing as mp
import os
import queue
import threading
import time
import numpy as np
from scipy.io import wavfile

from src.api.api import ServiceNames, TaskStatus, VoiceCloneProgress


from src.easevoice.inference import InferenceResult, InferenceTask, InferenceTaskData, Runner
from src.logger import logger
from src.utils.response import EaseVoiceResponse, ResponseStatus


class VoiceCloneService:
    """
    VoiceService is a long run service that listens for voice clone tasks and processes them.
    """

    def __init__(self):
        self.queue = mp.Queue()
        self.runner_process = None
        # self.runner_process = mp.Process(target=VoiceCloneService._init_runner, args=(self.queue,))
        # self.runner_process.start()

        # self._done = False
        # self._run_tasks = threading.Thread(target=self._run, daemon=True)
        # self._run_tasks.start()

    def start(self):
        if self.runner_process is not None:
            logger.info("Runner process already started")
            return
        self.runner_process = mp.Process(target=VoiceCloneService._init_runner, args=(self.queue,))
        self.runner_process.start()

    def close(self):
        if self.runner_process is not None:
            self.queue.put(1)
            self.runner_process.join()
            self.runner_process = None

    @staticmethod
    def _init_runner(queue: mp.Queue):
        """
        Call this method to start the runner process
        """
        runner = Runner(queue)
        runner.run()
        print("Runner process exited")
        gc.collect()

    def clone(self, params: dict):
        try:
            data = InferenceTaskData(**params)
            queue = mp.Queue()
            infer_task = InferenceTask(result_queue=queue, data=data)
            self.queue.put(infer_task)
            result: InferenceResult = infer_task.result_queue.get(timeout=600)
        except Exception as e:
            logger.error(f"failed to clone voice for {params}, error: {e}", exc_info=True)
            result = InferenceResult(error=str(e))

            if result.error:
                logger.error(f"failed to clone voice for {params}, error: {result.error}")
                return EaseVoiceResponse(ResponseStatus.FAILED, result.error)
            else:
                try:
                    sampling_rate = result.items[0][0]
                    audio = np.concatenate([item[1] for item in result.items])
                    return EaseVoiceResponse(ResponseStatus.SUCCESS, "Voice cloned successfully", {"sampling_rate": sampling_rate, "audio": audio})
                except Exception as e:
                    logger.error(f"failed to clone voice for {params}, error: {e}", exc_info=True)
                    return EaseVoiceResponse(ResponseStatus.FAILED, "failed to clone voice")
