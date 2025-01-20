import torch
import random
import os
import numpy as np
from ...logger import logger


def str2bool(val: str):
    return val.lower() in ["yes", "y", "true", "t", "1"]


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
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.enabled = True
            # 开启后会影响精度
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    except Exception as e:
        logger.error(f"Set seed failed: {e}")
    logger.info(f"Set seed to {seed}")
    return seed
