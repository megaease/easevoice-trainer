import dataclasses
import multiprocessing
from typing import Optional, Union
import random
import os
import logging

from src.utils.path import get_base_path
from src.logger import logger


from .tts import TTSConfig, TTS

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)


@dataclasses.dataclass
class InferenceTaskData:
    """
    Data for inference
    """
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    text_split_method: str
    aux_ref_audio_paths: list = dataclasses.field(default_factory=list)
    seed: int = -1
    top_k: int = 5
    top_p: int = 1
    temperature: float = 1.0
    batch_size: int = 20
    speed_factor: float = 1.0
    ref_text_free: bool = False
    split_bucket: bool = True
    fragment_interval: float = 0.3
    keep_random: bool = True
    parallel_infer: bool = True
    repetition_penalty: float = 1.3
    sovits_path: str = ""
    gpt_path: str = ""


class Runner:
    """
    Start Runner in a separate process
    Put InferenceTask into the queue
    Put int to stop the process
    Wait InferenceResult from the queue
    """

    def __init__(self):
        tts_config = TTSConfig(os.path.join(get_base_path(), "configs", "tts_infer.yaml"))
        logger.info(f"tts config: {tts_config}")

        self.tts_config = tts_config
        self.tts_pipeline = TTS(tts_config)

    def inference(self, data: InferenceTaskData):
        # change weight based on task
        try:
            self.tts_pipeline.update_weights(data.sovits_path, data.gpt_path)
        except Exception as e:
            logger.error(f"failed to update weights: {e}")
            # change back to default weights
            self.tts_pipeline.update_weights("", "")
            raise e

        seed = -1 if data.keep_random else data.seed
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        inputs = {
            "text": data.text,
            "text_lang": data.text_lang,
            "ref_audio_path": data.ref_audio_path,
            "aux_ref_audio_paths": [item.name for item in data.aux_ref_audio_paths],
            "prompt_text": data.prompt_text if not data.ref_text_free else "",
            "prompt_lang": data.prompt_lang,
            "top_k": data.top_k,
            "top_p": data.top_p,
            "temperature": data.temperature,
            "text_split_method": data.text_split_method,
            "batch_size": int(data.batch_size),
            "speed_factor": float(data.speed_factor),
            "split_bucket": data.split_bucket,
            "return_fragment": False,
            "fragment_interval": data.fragment_interval,
            "seed": actual_seed,
            "parallel_infer": data.parallel_infer,
            "repetition_penalty": data.repetition_penalty,
        }
        items = []
        for item in self.tts_pipeline.run(inputs):
            items.append(item)
        return items, actual_seed
