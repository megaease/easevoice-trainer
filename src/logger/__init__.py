import logging
import os


def _get_logger():
    log_level = os.environ.get("EASEVOICE_LOG_LEVEL", "INFO")
    log_level = log_level.upper()

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


logger = _get_logger()
