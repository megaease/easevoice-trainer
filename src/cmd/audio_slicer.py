# !/usr/bin/env python
# -*- encoding=utf8 -*-

import sys

sys.path.append('.')
sys.path.append('..')

import argparse
import json
import traceback

from src.service.audio import AudioService, AudioSlicerParams
from src.utils.helper.connector import MultiProcessOutputConnector
from src.utils.response import EaseVoiceResponse, ResponseStatus


def main():
    connector = MultiProcessOutputConnector()
    try:
        parser = argparse.ArgumentParser(
            description="run normalize service",
        )
        parser.add_argument(
            "-c",
            "--config",
            type=argparse.FileType("r"),
            required=True,
        )
        args = parser.parse_args()
        config_content = args.config.read()
        args.config.close()
        config = json.loads(config_content)

        params = AudioSlicerParams(**config)
        service = AudioService(source_dir=params.source_dir, output_dir=params.output_dir)
        output = service.slicer(
            threshold=params.threshold,
            min_length=params.min_length,
            min_interval=params.min_interval,
            hop_size=params.hop_size,
            max_silent_kept=params.max_silent_kept,
            normalize_max=params.normalize_max,
            alpha_mix=params.alpha_mix,
        )
        connector.write_response(output)
    except Exception as e:
        print(traceback.format_exc(), e)
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.FAILED, message=f"{e}"))


if __name__ == "__main__":
    main()
