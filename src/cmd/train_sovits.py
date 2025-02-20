#!/usr/bin/env python
# -*- encoding=utf8 -*-

import sys

sys.path.append('.')
sys.path.append('..')

import argparse
from dataclasses import asdict
import json

from src.train.sovits import SovitsTrain, SovitsTrainParams
from src.utils.helper.connector import MultiProcessOutputConnector
from src.utils.response import EaseVoiceResponse, ResponseStatus


def main():
    connector = MultiProcessOutputConnector()
    try:
        parser = argparse.ArgumentParser(
            description="run train sovits"
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

        params = SovitsTrainParams(**config)
        train = SovitsTrain(params=params)
        output = train.train()
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.SUCCESS, message="Finish train sovits", data=asdict(output)))
    except Exception as e:
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.FAILED, message=f"failed to train sovits, {e}"))


if __name__ == "__main__":
    main()
