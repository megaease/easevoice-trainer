import dataclasses
import multiprocessing
from typing import Optional, Union
import random
import os
import logging

from ...utils.config.config import GlobalCFG
from ...logger import logger


from .tts import TTSConfig, TTS

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)


@dataclasses.dataclass
class InferenceResult:
    items: list = []
    seed: int = -1
    error: Optional[str] = None


@dataclasses.dataclass
class InferenceTask:
    text: str
    text_lang: str
    ref_audio_path: str
    prompt_text: str
    prompt_lang: str
    text_split_method: str
    aux_ref_audio_paths: list = []
    seed = -1
    top_k = 5
    top_p = 1
    temperature = 1
    batch_size = 20
    speed_factor = 1.0
    ref_text_free = False
    split_bucket = True
    fragment_interval = 0.3
    keep_random = True
    parallel_infer = True
    repetition_penalty = 1.3


class Runner:
    """
    Start Runner in a separate process
    Put InferenceTask into the queue
    Put int to stop the process
    Wait InferenceResult from the queue
    """

    def __init__(self, queue: multiprocessing.Queue):
        cfg = GlobalCFG()

        gpt_path = os.environ.get("gpt_path", None)
        sovits_path = os.environ.get("sovits_path", None)
        cnhubert_base_path = os.environ.get("cnhubert_base_path", None)
        bert_path = os.environ.get("bert_path", None)

        tts_config = TTSConfig("GPT_SoVITS/configs/tts_infer.yaml")
        tts_config.device = cfg.device
        tts_config.is_half = cfg.is_half
        tts_config.t2s_weights_path = gpt_path if gpt_path is not None else tts_config.t2s_weights_path
        tts_config.vits_weights_path = sovits_path if sovits_path is not None else tts_config.vits_weights_path
        tts_config.cnhuhbert_base_path = cnhubert_base_path if cnhubert_base_path is not None else tts_config.cnhuhbert_base_path
        tts_config.bert_base_path = bert_path if bert_path is not None else tts_config.bert_base_path
        logger.info(f"tts config: {tts_config}")

        self.tts_config = tts_config
        self.tts_pipeline = TTS(tts_config)
        self.task_queue = queue
        self.done = False

    def run(self):
        while not self.done:
            task: Union[InferenceTask, int] = self.task_queue.get()
            if isinstance(task, int):
                logger.info("Received stop signal")
                return
            else:
                try:
                    items, seed = self._inference(task)
                    self.task_queue.put(
                        InferenceResult(items=items, seed=seed)
                    )
                except Exception as e:
                    logger.error(f"error: {e}")
                    self.task_queue.put(InferenceResult(error=str(e)))

    def _inference(self, task: InferenceTask):
        seed = -1 if task.keep_random else task.seed
        actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
        inputs = {
            "text": task.text,
            "text_lang": task.text_lang,
            "ref_audio_path": task.ref_audio_path,
            "aux_ref_audio_paths": [item.name for item in task.aux_ref_audio_paths],
            "prompt_text": task.prompt_text if not task.ref_text_free else "",
            "prompt_lang": task.prompt_lang,
            "top_k": task.top_k,
            "top_p": task.top_p,
            "temperature": task.temperature,
            "text_split_method": task.text_split_method,
            "batch_size": int(task.batch_size),
            "speed_factor": float(task.speed_factor),
            "split_bucket": task.split_bucket,
            "return_fragment": False,
            "fragment_interval": task.fragment_interval,
            "seed": actual_seed,
            "parallel_infer": task.parallel_infer,
            "repetition_penalty": task.repetition_penalty,
        }
        items = []
        for item in self.tts_pipeline.run(inputs):
            items.append(item)
        return items, actual_seed
