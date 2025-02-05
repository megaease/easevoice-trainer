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


class VoiceCloneService:
    """
    VoiceService is a long run service that listens for voice clone tasks and processes them.
    """

    def __init__(self):
        self.task_service = task_service

        self.queue = mp.Queue()
        self.runner_process = mp.Process(target=VoiceCloneService._init_runner, args=(self.queue,))
        self.runner_process.start()

        self._done = False
        self._run_tasks = threading.Thread(target=self._run, daemon=True)
        self._run_tasks.start()

    def close(self):
        self._done = True
        self.queue.put(1)
        self.runner_process.join()

    @staticmethod
    def _init_runner(queue: mp.Queue):
        """
        Call this method to start the runner process
        """
        runner = Runner(queue)
        runner.run()
        print("Runner process exited")
        gc.collect()

    def _run(self):
        while True:
            if self._done:
                logger.info("Voice clone service is shutting down")
                return

            tasks = self.task_service.filter_tasks(lambda t: t.service_name == ServiceNames.VOICE_CLONE and t.progress.status == TaskStatus.PENDING)
            if len(tasks) == 0:
                logger.debug("No pending tasks found for voice clone")
            else:
                task = tasks[0]
                logger.info(f"Processing task {task.taskID}, args: {task.args}")

                task.progress.status = TaskStatus.IN_PROGRESS
                self.task_service.submit_task(task)

                try:
                    data = InferenceTaskData(**task.args)
                    queue = mp.Queue()
                    infer_task = InferenceTask(result_queue=queue, data=data)
                    self.queue.put(infer_task)
                    result: InferenceResult = infer_task.result_queue.get(timeout=600)
                except Exception as e:
                    logger.error(f"failed to clone voice for {task.args}, error: {e}", exc_info=True)
                    result = InferenceResult(error=str(e))

                if result.error:
                    progress = VoiceCloneProgress(status=TaskStatus.FAILED, current_step="Failed", total_steps=1, completed_steps=1, current_step_progress=100, message=result.error)
                    task.progress = progress
                    self.task_service.submit_task(task)
                    logger.error(f"failed to clone voice for {task.args}, error: {result.error}")
                else:
                    try:
                        sampling_rate = result.items[0][0]
                        audio = np.concatenate([item[1] for item in result.items])
                        output_file = os.path.join(task.homePath, "output.wav")
                        wavfile.write(output_file, sampling_rate, audio)

                        progress = VoiceCloneProgress(status=TaskStatus.COMPLETED, current_step="Completed", total_steps=1, completed_steps=1, current_step_progress=100)
                        task.progress = progress
                        self.task_service.submit_task(task)
                        logger.info(f"Successfully cloned voice for {task.args}")

                    except Exception as e:
                        logger.error(f"failed to clone voice for {task.args}, error: {e}", exc_info=True)
                        progress = VoiceCloneProgress(status=TaskStatus.FAILED, current_step="Failed", total_steps=1, completed_steps=1, current_step_progress=100, message=str(e))
                        task.progress = progress
                        self.task_service.submit_task(task)
                        logger.error(f"failed to clone voice for {task.args}, error: {e}")

            time.sleep(1)
