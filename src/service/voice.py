import base64
from concurrent.futures import thread
from enum import Enum
import gc
import io
import multiprocessing as mp
import os
import queue
import threading
import time
import numpy as np
from scipy.io import wavfile
import soundfile as sf

from src.api.api import ServiceNames, TaskStatus, VoiceCloneProgress


from src.easevoice.inference import InferenceResult, InferenceTask, InferenceTaskData, Runner
from src.logger import logger
from src.utils.response import EaseVoiceResponse, ResponseStatus

class VoiceCloneStatus(Enum):
    RUNNING = "Running"
    COMPLETED = "Completed"
    ERROR = "Error"

class VoiceCloneService:
    """
    VoiceService is a long run service that listens for voice clone tasks and processes them.
    """

    def __init__(self):
        self.queue = mp.Queue()
        self.runner_process = mp.Process(target=VoiceCloneService._init_runner, args=(self.queue,))
        self.runner_process.start()

    def close(self):
        if self.runner_process is not None:
            self.queue.put(1)
            self.runner_process.terminate()
            self.runner_process.join(timeout=10)
            self.runner_process = None
    
    def get_status(self):
        if self.runner_process is None:
            return VoiceCloneStatus.COMPLETED
        elif self.runner_process.is_alive():
            return VoiceCloneStatus.RUNNING
        else:
            return VoiceCloneStatus.ERROR

    @staticmethod
    def _init_runner(queue: mp.Queue):
        """
        Call this method to start the runner process
        """
        runner = Runner(queue)
        runner.run()
        print("Voice clone runner process exited")
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
                    data = np.concatenate([item[1] for item in result.items])
                    buffer = io.BytesIO()
                    sf.write(buffer, data, sampling_rate, format="WAV")
                    audio = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    return EaseVoiceResponse(ResponseStatus.SUCCESS, "Voice cloned successfully", {"sampling_rate": sampling_rate, "audio": audio})
                except Exception as e:
                    logger.error(f"failed to clone voice for {params}, error: {e}", exc_info=True)
                    return EaseVoiceResponse(ResponseStatus.FAILED, "failed to clone voice")
