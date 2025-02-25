import base64
import gc
import io
import numpy as np
import soundfile as sf
import torch
import os

from src.easevoice.inference import InferenceTaskData, Runner
from src.logger import logger
from src.service.session import SessionManager
from src.train.helper import generate_random_name, list_train_gpts, list_train_sovits
from src.utils.response import EaseVoiceResponse, ResponseStatus


class VoiceCloneService:
    """
    VoiceService is a long run service that listens for voice clone tasks and processes them.
    """

    def __init__(self, session_manager: SessionManager):
        self.runner = Runner()
        self.session_manager = session_manager

    def close(self):
        if self.runner is not None:
            self.runner = None
            gc.collect()
            torch.cuda.empty_cache()

    def clone(self, uuid: str, params: dict):
        task = InferenceTaskData(**params)
        task = self._update_task_path(task)

        self.session_manager.update_session_info(uuid, {"message": "voice clone started"})
        items, seed = self.runner.inference(task)  # pyright: ignore
        self.session_manager.update_session_info(uuid, {"message": "voice clone completed, start to write audio"})

        sampling_rate = items[0][0]
        data = np.concatenate([item[1] for item in items])
        buffer = io.BytesIO()
        sf.write(buffer, data, sampling_rate, format="WAV")

        filename = "voice_" + generate_random_name()
        os.makedirs(task.output_dir, exist_ok=True)
        path = os.path.join(task.output_dir, f"{filename}.wav")
        with open(path, "wb") as f:
            f.write(buffer.getvalue())
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Voice cloned successfully", {"sampling_rate": sampling_rate, "output_path": path})

    def _update_task_path(self, data: InferenceTaskData):
        if data.gpt_path == "default":
            data.gpt_path = ""
        if data.sovits_path == "default":
            data.sovits_path = ""

        if data.gpt_path != "":
            gpts = list_train_gpts()
            if data.gpt_path in gpts:
                data.gpt_path = gpts[data.gpt_path]
            else:
                logger.error(f"failed to find gpt model for {data.gpt_path}")
                raise ValueError(f"failed to find gpt model for {data.gpt_path}")
        if data.sovits_path != "":
            sovits = list_train_sovits()
            if data.sovits_path in sovits:
                data.sovits_path = sovits[data.sovits_path]
            else:
                logger.error(f"failed to find sovits model for {data.sovits_path}")
                raise ValueError(f"failed to find sovits model for {data.sovits_path}")
        return data
