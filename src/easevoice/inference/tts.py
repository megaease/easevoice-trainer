
from pydantic import BaseModel

from src.utils.config.config import GlobalCFG
from .preprocessor import TextPreprocessor
from .segmentation import SPLITS
from src.easevoice.module.mel_processing import spectrogram_torch
from src.utils.audio import load_audio
from time import time as ttime
import librosa
from src.easevoice.module.models import SynthesizerTrn
from src.easevoice.feature_extractor.cnhubert import CNHubert
from src.easevoice.soundstorm.auto_reg.models.t2s_lightning_module import Text2SemanticLightningModule
from transformers import AutoModelForMaskedLM, AutoTokenizer
import yaml
import torch
import numpy as np
from typing import List, Optional, Tuple, Union
import os
import ffmpeg
from copy import deepcopy
import math
import os
import sys
import gc
import random
import traceback
from src.logger import logger

from tqdm import tqdm

sys.path.append(os.getcwd())


def set_seed(seed: int):
    seed = int(seed)
    seed = seed if seed != -1 else random.randrange(1 << 32)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception as e:
        logger.error(f"Set seed failed: {e}")
    logger.info(f"Set seed to {seed}")
    return seed


def _get_default_configs():
    global_config = GlobalCFG()
    return {
        "device": global_config.device,
        "is_half": global_config.is_half,
        "t2s_weights_path": global_config.gpt_path,
        "vits_weights_path": global_config.sovits_path,
        "cnhuhbert_base_path": global_config.cnhubert_path,
        "bert_base_path": global_config.bert_path,
    }


class TTSConfig:
    default_configs = {
        "default": _get_default_configs(),
    }

    # "all_zh": all chinese
    # "en": all english
    # "all_ja": all japanese
    # "all_yue": all yue chinese
    # "all_ko": all korean
    # "zh": chinese and english
    # "ja": japanese and english
    # "yue": yue and english
    # "ko": korean and english
    # "auto": multi languages
    # "auto_yue": multi languages with yue
    languages: list = ["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"]

    def __init__(self, configs: Union[dict, str, None] = None):  # pyright: ignore
        global_config = GlobalCFG()

        configs_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        os.makedirs(configs_base_path, exist_ok=True)
        self.configs_path: str = os.path.join(configs_base_path, "tts_infer.yaml")

        if configs in ["", None]:
            if not os.path.exists(self.configs_path):
                self.save_configs()
                logger.info(f"Create default config file at {self.configs_path}")
            configs: dict = deepcopy(self.default_configs)

        if isinstance(configs, str):
            self.configs_path = configs
            configs: dict = self._load_configs(self.configs_path)

        assert isinstance(configs, dict)
        self.default_configs["default"] = configs.get("default", self.default_configs["default"])

        self.configs: dict = configs.get("custom", deepcopy(self.default_configs["default"]))
        self.device = self.configs.get("device", global_config.device)
        self.is_half = self.configs.get("is_half", global_config.is_half)

        def get_path(key: str):
            path = self.configs.get(key, None)
            if (path == "" or path is None) or (not os.path.exists(path)):
                logger.warning(f"{key} path {path} not exist or found, fall back to default.")
                return self.default_configs["default"][key]
            return path

        self.t2s_weights_path = get_path("t2s_weights_path")
        self.default_t2s_weights_path = global_config.gpt_path

        self.vits_weights_path = get_path("vits_weights_path")
        self.default_vits_weights_path = global_config.sovits_path

        self.bert_base_path = get_path("bert_base_path")
        self.cnhuhbert_base_path = get_path("cnhuhbert_base_path")

        self.update_configs()

        self.max_sec = None
        self.hz: int = 50
        self.semantic_frame_rate: str = "25hz"
        self.segment_size: int = 20480
        self.filter_length: int = 2048
        self.sampling_rate: int = 32000
        self.hop_length: int = 640
        self.win_length: int = 2048
        self.n_speakers: int = 300

    def _load_configs(self, configs_path: str) -> dict:
        if not os.path.exists(configs_path):
            logger.warning(f"Config file not found at {configs_path}, use default configs.")
            self.save_configs(configs_path)
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        return configs

    def save_configs(self, configs_path: Optional[str] = None) -> None:
        configs = deepcopy(self.default_configs)
        if self.configs is not None:
            configs["custom"] = self.update_configs()

        if configs_path is None:
            configs_path = self.configs_path
        with open(configs_path, 'w') as f:
            yaml.dump(configs, f)

    def update_configs(self):
        self.config = {
            "device": str(self.device),
            "is_half": self.is_half,
            "t2s_weights_path": self.t2s_weights_path,
            "vits_weights_path": self.vits_weights_path,
            "bert_base_path": self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
        }
        return self.config

    def __str__(self):
        self.configs = self.update_configs()
        string = "TTS Config".center(100, '-') + '\n'
        for k, v in self.configs.items():
            string += f"{str(k).ljust(20)}: {str(v)}\n"
        string += "-" * 100 + '\n'
        return string

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.configs_path)

    def __eq__(self, other):
        return isinstance(other, TTSConfig) and self.configs_path == other.configs_path


class TTS:
    def __init__(self, configs: Union[dict, str, TTSConfig]):
        if isinstance(configs, TTSConfig):
            self.configs = configs
        else:
            self.configs: TTSConfig = TTSConfig(configs)

        self.t2s_model: Text2SemanticLightningModule = None  # pyright: ignore
        self.vits_model: SynthesizerTrn = None  # pyright: ignore
        self.bert_tokenizer: AutoTokenizer = None  # pyright: ignore
        self.bert_model: AutoModelForMaskedLM = None  # pyright: ignore
        self.cnhuhbert_model: CNHubert = None  # pyright: ignore

        self.current_sovits_path = configs.vits_weights_path  # pyright: ignore
        self.current_gpt_path = configs.t2s_weights_path  # pyright: ignore

        self._init_models()

        self.text_preprocessor: TextPreprocessor = \
            TextPreprocessor(self.bert_model,
                             self.bert_tokenizer,
                             self.configs.device)

        self.prompt_cache: dict = {
            "ref_audio_path": None,
            "prompt_semantic": None,
            "refer_spec": [],
            "prompt_text": None,
            "prompt_lang": None,
            "phones": None,
            "bert_features": None,
            "norm_text": None,
            "aux_ref_audio_paths": [],
        }

        self.stop_flag: bool = False
        self.precision: torch.dtype = torch.float16 if self.configs.is_half else torch.float32

    def update_weights(self, sovits_path: str, gpt_path: str):
        if sovits_path != self.current_sovits_path:
            if sovits_path == "":
                # empty sovits path, use default path
                if self.current_sovits_path != self.configs.default_vits_weights_path:
                    self.current_sovits_path = self.configs.default_vits_weights_path
                    self.init_vits_weights(self.current_sovits_path)
            else:
                self.current_sovits_path = sovits_path
                self.init_vits_weights(self.current_sovits_path)

        if gpt_path != self.current_gpt_path:
            if gpt_path == "":
                # empty gpt path, use default path
                if self.current_gpt_path != self.configs.default_t2s_weights_path:
                    self.current_gpt_path = self.configs.default_t2s_weights_path
                    self.init_t2s_weights(self.current_gpt_path)
            else:
                self.current_gpt_path = gpt_path
                self.init_t2s_weights(self.current_gpt_path)

    def _init_models(self,):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)

    def init_cnhuhbert_weights(self, base_path: str):
        logger.info(f"Loading CNHuBERT weights from {base_path}")
        self.cnhuhbert_model = CNHubert(base_path)
        self.cnhuhbert_model = self.cnhuhbert_model.eval()
        self.cnhuhbert_model = self.cnhuhbert_model.to(self.configs.device)
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.cnhuhbert_model = self.cnhuhbert_model.half()

    def init_bert_weights(self, base_path: str):
        logger.info(f"Loading BERT weights from {base_path}")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)  # pyright: ignore
        self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
        self.bert_model = self.bert_model.eval()  # pyright: ignore
        self.bert_model = self.bert_model.to(self.configs.device)  # pyright: ignore
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.bert_model = self.bert_model.half()  # pyright: ignore

    def init_vits_weights(self, weights_path: str):
        logger.info(f"Loading VITS weights from {weights_path}")
        self.configs.vits_weights_path = weights_path
        dict_s2 = torch.load(weights_path, map_location=self.configs.device)
        hps = dict_s2["config"]
        if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
            raise ValueError("The model is version v1, please use the latest version model.")
        self.configs.save_configs()

        if isinstance(hps, BaseModel):
            hps = hps.model_dump()
        self.configs.filter_length = hps["data"]["filter_length"]
        self.configs.segment_size = hps["train"]["segment_size"]
        self.configs.sampling_rate = hps["data"]["sampling_rate"]
        self.configs.hop_length = hps["data"]["hop_length"]
        self.configs.win_length = hps["data"]["win_length"]
        self.configs.n_speakers = hps["data"]["n_speakers"]
        self.configs.semantic_frame_rate = "25hz"
        kwargs = hps["model"]
        vits_model = SynthesizerTrn(
            self.configs.filter_length // 2 + 1,
            self.configs.segment_size // self.configs.hop_length,
            n_speakers=self.configs.n_speakers,
            **kwargs
        )

        if hasattr(vits_model, "enc_q"):
            del vits_model.enc_q

        vits_model = vits_model.to(self.configs.device)
        vits_model = vits_model.eval()
        vits_model.load_state_dict(dict_s2["weight"], strict=False)
        self.vits_model = vits_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.vits_model = self.vits_model.half()

    def init_t2s_weights(self, weights_path: str):
        logger.info(f"Loading Text2Semantic weights from {weights_path}")
        self.configs.t2s_weights_path = weights_path
        self.configs.save_configs()
        self.configs.hz = 50
        dict_s1 = torch.load(weights_path, map_location=self.configs.device)
        config = dict_s1["config"]
        self.configs.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(self.configs.device)
        t2s_model = t2s_model.eval()
        self.t2s_model = t2s_model
        if self.configs.is_half and str(self.configs.device) != "cpu":
            self.t2s_model = self.t2s_model.half()

    def enable_half_precision(self, enable: bool = True, save: bool = True):
        '''
            To enable half precision for the TTS model.
            Args:
                enable: bool, whether to enable half precision.

        '''
        if str(self.configs.device) == "cpu" and enable:
            logger.info("Half precision is not supported on CPU.")
            return

        self.configs.is_half = enable
        self.precision = torch.float16 if enable else torch.float32
        if save:
            self.configs.save_configs()
        if enable:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.half()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.half()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.half()  # pyright: ignore
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.half()
        else:
            if self.t2s_model is not None:
                self.t2s_model = self.t2s_model.float()
            if self.vits_model is not None:
                self.vits_model = self.vits_model.float()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.float()  # pyright: ignore
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.float()

    def set_device(self, device: torch.device, save: bool = True):
        '''
            To set the device for all models.
            Args:
                device: torch.device, the device to use for all models.
        '''
        self.configs.device = device
        if save:
            self.configs.save_configs()
        if self.t2s_model is not None:
            self.t2s_model = self.t2s_model.to(device)
        if self.vits_model is not None:
            self.vits_model = self.vits_model.to(device)
        if self.bert_model is not None:
            self.bert_model = self.bert_model.to(device)  # pyright: ignore
        if self.cnhuhbert_model is not None:
            self.cnhuhbert_model = self.cnhuhbert_model.to(device)

    def set_ref_audio(self, ref_audio_path: str):
        '''
            To set the reference audio for the TTS model,
                including the prompt_semantic and refer_spepc.
            Args:
                ref_audio_path: str, the path of the reference audio.
        '''
        self._set_prompt_semantic(ref_audio_path)
        self._set_ref_spec(ref_audio_path)
        self._set_ref_audio_path(ref_audio_path)

    def _set_ref_audio_path(self, ref_audio_path):
        self.prompt_cache["ref_audio_path"] = ref_audio_path

    def _set_ref_spec(self, ref_audio_path):
        spec = self._get_ref_spec(ref_audio_path)
        if self.prompt_cache["refer_spec"] in [[], None]:
            self.prompt_cache["refer_spec"] = [spec]
        else:
            self.prompt_cache["refer_spec"][0] = spec

    def _get_ref_spec(self, ref_audio_path):
        audio = load_audio(ref_audio_path, int(self.configs.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if (maxx > 1):
            audio /= min(2, maxx)  # pyright: ignore
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.configs.filter_length,
            self.configs.sampling_rate,
            self.configs.hop_length,
            self.configs.win_length,
            center=False,
        )
        spec = spec.to(self.configs.device)
        if self.configs.is_half:
            spec = spec.half()
        return spec

    def _set_prompt_semantic(self, ref_wav_path: str):
        zero_wav = np.zeros(
            int(self.configs.sampling_rate * 0.3),
            dtype=np.float16 if self.configs.is_half else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                raise OSError("audio length should be in 3~10 seconds.")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            wav16k = wav16k.to(self.configs.device)
            zero_wav_torch = zero_wav_torch.to(self.configs.device)
            if self.configs.is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()

            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))[ # pyright: ignore
                "last_hidden_state"
            ].transpose(
                1, 2
            )
            codes = self.vits_model.extract_latent(hubert_feature)

            prompt_semantic = codes[0, 0].to(self.configs.device)
            self.prompt_cache["prompt_semantic"] = prompt_semantic

    def batch_sequences(self, sequences: List[torch.Tensor], axis: int = 0, pad_value: int = 0, max_length: Optional[int] = None):
        seq = sequences[0]
        ndim = seq.dim()
        if axis < 0:
            axis += ndim
        dtype: torch.dtype = seq.dtype
        pad_value = torch.tensor(pad_value, dtype=dtype)  # pyright: ignore
        seq_lengths = [seq.shape[axis] for seq in sequences]
        if max_length is None:
            max_length = max(seq_lengths)
        else:
            max_length = max(seq_lengths) if max_length < max(seq_lengths) else max_length

        padded_sequences = []
        for seq, length in zip(sequences, seq_lengths):
            padding = [0] * axis + [0, max_length - length] + [0] * (ndim - axis - 1)
            padded_seq = torch.nn.functional.pad(seq, padding, value=pad_value)
            padded_sequences.append(padded_seq)
        batch = torch.stack(padded_sequences)
        return batch

    def to_batch(self, data: list,
                 prompt_data: Optional[dict] = None,
                 batch_size: int = 5,
                 threshold: float = 0.75,
                 split_bucket: bool = True,
                 device: torch.device = torch.device("cpu"),
                 precision: torch.dtype = torch.float32,
                 ):
        _data: list = []
        index_and_len_list = []
        for idx, item in enumerate(data):
            norm_text_len = len(item["norm_text"])
            index_and_len_list.append([idx, norm_text_len])

        batch_index_list = []
        if split_bucket:
            index_and_len_list.sort(key=lambda x: x[1])
            index_and_len_list = np.array(index_and_len_list, dtype=np.int64)

            batch_index_list_len = 0
            pos = 0
            while pos < index_and_len_list.shape[0]:
                pos_end = min(pos+batch_size, index_and_len_list.shape[0])
                while pos < pos_end:
                    batch = index_and_len_list[pos:pos_end, 1].astype(np.float32)
                    score = batch[(pos_end-pos)//2]/(batch.mean()+1e-8)
                    if (score >= threshold) or (pos_end-pos == 1):
                        batch_index = index_and_len_list[pos:pos_end, 0].tolist()
                        batch_index_list_len += len(batch_index)
                        batch_index_list.append(batch_index)
                        pos = pos_end
                        break
                    pos_end = pos_end-1

            assert batch_index_list_len == len(data)

        else:
            for i in range(len(data)):
                if i % batch_size == 0:
                    batch_index_list.append([])
                batch_index_list[-1].append(i)

        for batch_idx, index_list in enumerate(batch_index_list):
            item_list = [data[idx] for idx in index_list]
            phones_list = []
            phones_len_list = []
            all_phones_list = []
            all_phones_len_list = []
            all_bert_features_list = []
            norm_text_batch = []
            all_bert_max_len = 0
            all_phones_max_len = 0
            for item in item_list:
                if prompt_data is not None:
                    all_bert_features = torch.cat([prompt_data["bert_features"], item["bert_features"]], 1)\
                        .to(dtype=precision, device=device)
                    all_phones = torch.LongTensor(prompt_data["phones"]+item["phones"]).to(device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                else:
                    all_bert_features = item["bert_features"]\
                        .to(dtype=precision, device=device)
                    phones = torch.LongTensor(item["phones"]).to(device)
                    all_phones = phones

                all_bert_max_len = max(all_bert_max_len, all_bert_features.shape[-1])
                all_phones_max_len = max(all_phones_max_len, all_phones.shape[-1])

                phones_list.append(phones)
                phones_len_list.append(phones.shape[-1])
                all_phones_list.append(all_phones)
                all_phones_len_list.append(all_phones.shape[-1])
                all_bert_features_list.append(all_bert_features)
                norm_text_batch.append(item["norm_text"])

            phones_batch = phones_list
            all_phones_batch = all_phones_list
            all_bert_features_batch = all_bert_features_list

            max_len = max(all_bert_max_len, all_phones_max_len)

            batch = {
                "phones": phones_batch,
                "phones_len": torch.LongTensor(phones_len_list).to(device),
                "all_phones": all_phones_batch,
                "all_phones_len": torch.LongTensor(all_phones_len_list).to(device),
                "all_bert_features": all_bert_features_batch,
                "norm_text": norm_text_batch,
                "max_len": max_len,
            }
            _data.append(batch)

        return _data, batch_index_list

    def recovery_order(self, data: list, batch_index_list: list) -> list:
        '''
        Recovery the order of the audio according to the batch_index_list.

        Args:
            data (List[list(np.ndarray)]): the out of order audio .
            batch_index_list (List[list[int]]): the batch index list.

        Returns:
            list (List[np.ndarray]): the data in the original order.
        '''
        length = len(sum(batch_index_list, []))
        _data = [None]*length
        for i, index_list in enumerate(batch_index_list):
            for j, index in enumerate(index_list):
                _data[index] = data[i][j]
        return _data

    def stop(self,):
        '''
        Stop the inference process.
        '''
        self.stop_flag = True

    @torch.no_grad()
    def run(self, inputs: dict):
        """
        Text to speech inference.

        Args:
            inputs (dict):
                {
                    "text": "",                   # str.(required) text to be synthesized
                    "text_lang: "",               # str.(required) language of the text to be synthesized
                    "ref_audio_path": "",         # str.(required) reference audio path
                    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
                    "prompt_text": "",            # str.(optional) prompt text for the reference audio
                    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                    "top_k": 5,                   # int. top k sampling
                    "top_p": 1,                   # float. top p sampling
                    "temperature": 1,             # float. temperature for sampling
                    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
                    "batch_size": 1,              # int. batch size for inference
                    "batch_threshold": 0.75,      # float. threshold for batch splitting.
                    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                    "return_fragment": False,     # bool. step by step return the audio fragment.
                    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                    "seed": -1,                   # int. random seed for reproducibility.
                    "parallel_infer": True,       # bool. whether to use parallel inference.
                    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
                }
        returns:
            Tuple[int, np.ndarray]: sampling rate and audio data.
        """
        ########## variables initialization ###########
        self.stop_flag: bool = False
        text: str = inputs.get("text", "")
        text_lang: str = inputs.get("text_lang", "")
        ref_audio_path: str = inputs.get("ref_audio_path", "")
        aux_ref_audio_paths: list = inputs.get("aux_ref_audio_paths", [])
        prompt_text: str = inputs.get("prompt_text", "")
        prompt_lang: str = inputs.get("prompt_lang", "")
        top_k: int = inputs.get("top_k", 5)
        top_p: float = inputs.get("top_p", 1)
        temperature: float = inputs.get("temperature", 1)
        text_split_method: str = inputs.get("text_split_method", "cut0")
        batch_size = inputs.get("batch_size", 1)
        batch_threshold = inputs.get("batch_threshold", 0.75)
        speed_factor = inputs.get("speed_factor", 1.0)
        split_bucket = inputs.get("split_bucket", True)
        return_fragment = inputs.get("return_fragment", False)
        fragment_interval = inputs.get("fragment_interval", 0.3)
        seed = inputs.get("seed", -1)
        seed = -1 if seed in ["", None] else seed
        actual_seed = set_seed(seed)
        parallel_infer = inputs.get("parallel_infer", True)
        repetition_penalty = inputs.get("repetition_penalty", 1.35)

        logger.info(f"text: {text}, parallel infer: {parallel_infer}, return fragment: {return_fragment}, split bucket: {split_bucket}")
        if parallel_infer:
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_batch_infer  # pyright: ignore
        else:
            self.t2s_model.model.infer_panel = self.t2s_model.model.infer_panel_naive_batched  # pyright: ignore

        if return_fragment:
            if split_bucket:
                split_bucket = False

        if split_bucket and speed_factor == 1.0:
            pass
        elif speed_factor != 1.0:
            split_bucket = False

        if fragment_interval < 0.01:
            fragment_interval = 0.01
            logger.info("fragment interval is too small, set to 0.01")

        no_prompt_text = False
        if prompt_text in [None, ""]:
            no_prompt_text = True

        assert text_lang in self.configs.languages
        if not no_prompt_text:
            assert prompt_lang in self.configs.languages

        if ref_audio_path in [None, ""] and \
                ((self.prompt_cache["prompt_semantic"] is None) or (self.prompt_cache["refer_spec"] in [None, []])):
            raise ValueError("ref_audio_path cannot be empty")

        ###### setting reference audio and prompt text preprocessing ########
        t0 = ttime()
        if (ref_audio_path is not None) and (ref_audio_path != self.prompt_cache["ref_audio_path"]):
            if not os.path.exists(ref_audio_path):
                raise ValueError(f"{ref_audio_path} not exists")
            self.set_ref_audio(ref_audio_path)

        aux_ref_audio_paths = aux_ref_audio_paths if aux_ref_audio_paths is not None else []
        paths = set(aux_ref_audio_paths) & set(self.prompt_cache["aux_ref_audio_paths"])
        if not (len(list(paths)) == len(aux_ref_audio_paths) == len(self.prompt_cache["aux_ref_audio_paths"])):
            self.prompt_cache["aux_ref_audio_paths"] = aux_ref_audio_paths
            self.prompt_cache["refer_spec"] = [self.prompt_cache["refer_spec"][0]]
            for path in aux_ref_audio_paths:
                if path in [None, ""]:
                    continue
                if not os.path.exists(path):
                    logger.info(f"audio file not exists, skip:{path}")
                    continue
                self.prompt_cache["refer_spec"].append(self._get_ref_spec(path))

        if not no_prompt_text:
            prompt_text = prompt_text.strip("\n")
            if (prompt_text[-1] not in SPLITS):
                prompt_text += "。" if prompt_lang != "en" else "."
            print(f"final prompt text is: {prompt_text}")
            if self.prompt_cache["prompt_text"] != prompt_text:
                self.prompt_cache["prompt_text"] = prompt_text
                self.prompt_cache["prompt_lang"] = prompt_lang
                phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(prompt_text, prompt_lang)
                self.prompt_cache["phones"] = phones
                self.prompt_cache["bert_features"] = bert_features
                self.prompt_cache["norm_text"] = norm_text

        ###### text preprocessing ########
        t1 = ttime()
        data: Optional[list] = None
        if not return_fragment:
            data = self.text_preprocessor.preprocess(text, text_lang, text_split_method)
            if len(data) == 0:
                yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                           dtype=np.int16)
                return

            batch_index_list: Optional[list] = None
            data, batch_index_list = self.to_batch(data,
                                                   prompt_data=self.prompt_cache if not no_prompt_text else None,
                                                   batch_size=batch_size,
                                                   threshold=batch_threshold,
                                                   split_bucket=split_bucket,
                                                   device=self.configs.device,
                                                   precision=self.precision
                                                   )
        else:
            logger.info("############ split text ############")
            texts = self.text_preprocessor.pre_seg_text(text, text_lang, text_split_method)
            data = []
            for i in range(len(texts)):
                if i % batch_size == 0:
                    data.append([])
                data[-1].append(texts[i])

            def make_batch(batch_texts):
                batch_data = []
                logger.info("############ get bert feature ############")
                for text in tqdm(batch_texts):
                    phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(text, text_lang)
                    if phones is None:
                        continue
                    res = {
                        "phones": phones,
                        "bert_features": bert_features,
                        "norm_text": norm_text,
                    }
                    batch_data.append(res)
                if len(batch_data) == 0:
                    return None
                batch, _ = self.to_batch(batch_data,
                                         prompt_data=self.prompt_cache if not no_prompt_text else None,
                                         batch_size=batch_size,
                                         threshold=batch_threshold,
                                         split_bucket=False,
                                         device=self.configs.device,
                                         precision=self.precision
                                         )
                return batch[0]

        t2 = ttime()
        try:
            logger.info("############ inference ############")
            ###### inference ######
            t_34 = 0.0
            t_45 = 0.0
            audio = []
            for item in data:
                t3 = ttime()
                if return_fragment:
                    item = make_batch(item)  # pyright: ignore
                    if item is None:
                        continue

                batch_phones: List[torch.LongTensor] = item["phones"]
                batch_phones_len: torch.LongTensor = item["phones_len"]
                all_phoneme_ids: torch.LongTensor = item["all_phones"]
                all_phoneme_lens: torch.LongTensor = item["all_phones_len"]
                all_bert_features: torch.LongTensor = item["all_bert_features"]
                norm_text: str = item["norm_text"]
                max_len = item["max_len"]

                print("前端处理后的文本(每句):", norm_text)
                if no_prompt_text:
                    prompt = None
                else:
                    prompt = self.prompt_cache["prompt_semantic"].expand(len(all_phoneme_ids), -1).to(self.configs.device)

                pred_semantic_list, idx_list = self.t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_lens,
                    prompt,  # pyright: ignore
                    all_bert_features,
                    top_k=top_k,
                    top_p=top_p,  # pyright: ignore
                    temperature=temperature,
                    early_stop_num=self.configs.hz * self.configs.max_sec,  # pyright: ignore
                    max_len=max_len,
                    repetition_penalty=repetition_penalty,
                )
                t4 = ttime()
                t_34 += t4 - t3

                refer_audio_spec: torch.Tensor = [item.to(dtype=self.precision, device=self.configs.device) for item in self.prompt_cache["refer_spec"]]  # pyright: ignore

                batch_audio_fragment = []

                if speed_factor == 1.0:
                    pred_semantic_list = [item[-idx:] for item, idx in zip(pred_semantic_list, idx_list)]  # pyright: ignore
                    upsample_rate = math.prod(self.vits_model.upsample_rates)
                    audio_frag_idx = [pred_semantic_list[i].shape[0]*2*upsample_rate for i in range(0, len(pred_semantic_list))]
                    audio_frag_end_idx = [sum(audio_frag_idx[:i+1]) for i in range(0, len(audio_frag_idx))]
                    all_pred_semantic = torch.cat(pred_semantic_list).unsqueeze(0).unsqueeze(0).to(self.configs.device)
                    _batch_phones = torch.cat(batch_phones).unsqueeze(0).to(self.configs.device)  # pyright: ignore
                    _batch_audio_fragment = (self.vits_model.decode(
                        all_pred_semantic, _batch_phones, refer_audio_spec, speed=speed_factor
                    ).detach()[0, 0, :])
                    audio_frag_end_idx.insert(0, 0)
                    batch_audio_fragment = [_batch_audio_fragment[audio_frag_end_idx[i-1]:audio_frag_end_idx[i]] for i in range(1, len(audio_frag_end_idx))]
                else:
                    for i, idx in enumerate(idx_list):  # pyright: ignore
                        phones = batch_phones[i].unsqueeze(0).to(self.configs.device)
                        _pred_semantic = (pred_semantic_list[i][-idx:].unsqueeze(0).unsqueeze(0)) 
                        audio_fragment = (self.vits_model.decode(
                            _pred_semantic, phones, refer_audio_spec, speed=speed_factor
                        ).detach()[0, 0, :])
                        batch_audio_fragment.append(
                            audio_fragment
                        )

                t5 = ttime()
                t_45 += t5 - t4
                if return_fragment:
                    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t4 - t3, t5 - t4))
                    yield self.audio_postprocess([batch_audio_fragment],  # pyright: ignore
                                                 self.configs.sampling_rate,
                                                 None,
                                                 speed_factor,
                                                 False,
                                                 fragment_interval
                                                 )
                else:
                    audio.append(batch_audio_fragment)

                if self.stop_flag:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                               dtype=np.int16)
                    return

            if not return_fragment:
                logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t_34, t_45))
                if len(audio) == 0:
                    yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                               dtype=np.int16)
                    return
                yield self.audio_postprocess(audio,
                                             self.configs.sampling_rate,
                                             batch_index_list,  # pyright: ignore
                                             speed_factor,
                                             split_bucket,
                                             fragment_interval
                                             )

        except Exception as e:
            traceback.print_exc()
            # must return an empty audio, otherwise it will cause the memory not to be released.
            yield self.configs.sampling_rate, np.zeros(int(self.configs.sampling_rate),
                                                       dtype=np.int16)
            # reset the model, otherwise it will cause the memory not to be released.
            del self.t2s_model
            del self.vits_model
            self.t2s_model = None  # pyright: ignore
            self.vits_model = None  # pyright: ignore
            self.init_t2s_weights(self.configs.t2s_weights_path)
            self.init_vits_weights(self.configs.vits_weights_path)
            raise e
        finally:
            self.empty_cache()

    def empty_cache(self):
        try:
            gc.collect()
            if "cuda" in str(self.configs.device):
                torch.cuda.empty_cache()
            elif str(self.configs.device) == "mps":
                torch.mps.empty_cache()
        except:
            pass

    def audio_postprocess(self,
                          audio: List[torch.Tensor],
                          sr: int,
                          batch_index_list: Optional[list] = None,
                          speed_factor: float = 1.0,
                          split_bucket: bool = True,
                          fragment_interval: float = 0.3
                          ) -> Tuple[int, np.ndarray]:
        zero_wav = torch.zeros(
            int(self.configs.sampling_rate * fragment_interval),
            dtype=self.precision,
            device=self.configs.device
        )

        for i, batch in enumerate(audio):
            for j, audio_fragment in enumerate(batch):
                # prevent 16bit overflow
                max_audio = torch.abs(audio_fragment).max() 
                if max_audio > 1:
                    audio_fragment /= max_audio
                audio_fragment: torch.Tensor = torch.cat([audio_fragment, zero_wav], dim=0)
                audio[i][j] = audio_fragment.cpu().numpy()

        if split_bucket:
            audio = self.recovery_order(audio, batch_index_list)  # pyright: ignore
        else:
            audio = sum(audio, [])  # pyright: ignore

        audio = np.concatenate(audio, 0)  # pyright: ignore
        audio = (audio * 32768).astype(np.int16)  # pyright: ignore
        return sr, audio  # pyright: ignore