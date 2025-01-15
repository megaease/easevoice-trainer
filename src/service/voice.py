import gc
from multiprocessing import Queue, Process
import threading


from ..easevoice.inference import InferenceResult, InferenceTask, Runner


class VoiceService:
    def __init__(self):
        self.queue = Queue()
        self.runner_process = Process(target=VoiceService.init_runner, args=(self.queue,))
        self.runner_process.start()
        self.lock = threading.Lock()

    @staticmethod
    def init_runner(queue: Queue):
        """
        Call this method to start the runner process
        """
        runner = Runner(queue)
        runner.run()
        gc.collect()

    def clone(self, input: dict):
        success = self.lock.acquire(blocking=False)
        if not success:
            raise Exception("inference task is running, please wait")
        task = InferenceTask(**input)
        self.queue.put(task)
        result: InferenceResult = self.queue.get()
        self.lock.release()
        return result
