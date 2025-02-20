import base64
import gc
import io
import numpy as np
import soundfile as sf
import torch


from src.easevoice.inference import InferenceTaskData, Runner
from src.logger import logger
from src.service.session import SessionManager
from src.train.helper import list_train_gpts, list_train_sovits
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
        data = InferenceTaskData(**params)
        data = self._update_task_path(data)

        self.session_manager.update_session_info(uuid, {"message": "voice clone started"})
        items, seed = self.runner.inference(data)  # pyright: ignore
        self.session_manager.update_session_info(uuid, {"message": "voice clone completed, start to write audio"})

        sampling_rate = items[0][0]
        data = np.concatenate([item[1] for item in items])
        buffer = io.BytesIO()
        sf.write(buffer, data, sampling_rate, format="WAV")
        audio = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Voice cloned successfully", {"sampling_rate": sampling_rate, "audio": audio})

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
