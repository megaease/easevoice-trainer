import gc
import multiprocessing as mp
import threading

from torch import mul

from ..utils.response import EaseVoiceResponse, ResponseStatus


from ..easevoice.inference import InferenceResult, InferenceTask, InferenceTaskData, Runner
from ..logger import logger


class VoiceService:
    def __init__(self):
        self.queue = mp.Queue()
        self.runner_process = mp.Process(target=VoiceService.init_runner, args=(self.queue,))
        self.runner_process.start()
        self.locker = threading.Lock()

    @staticmethod
    def init_runner(queue: mp.Queue):
        """
        Call this method to start the runner process
        """
        runner = Runner(queue)
        runner.run()
        gc.collect()

    def clone(self, input: dict):
        ok = self.locker.acquire(timeout=5)
        if not ok:
            return EaseVoiceResponse(ResponseStatus.FAILED, "There is another task running, please try again later")

        try:
            data = InferenceTaskData(**input)
            queue = mp.Queue()
            task = InferenceTask(result_queue=queue, data=data)
            self.queue.put(task)
            result: InferenceResult = task.result_queue.get(timeout=600)
        except Exception as e:
            logger.error(f"failed to clone voice for {input}, error: {e}", exc_info=True)
            result = InferenceResult(error=str(e))

        finally:
            self.locker.release()

        if result.error:
            return EaseVoiceResponse(ResponseStatus.FAILED, result.error)

        return EaseVoiceResponse(
            ResponseStatus.SUCCESS,
            "Cloned voice successfully",
            {
                "items": result.items,
                "seed": result.seed
            })
