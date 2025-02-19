#!/usr/bin/env python
# -*- encoding=utf8 -*-
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import json
import traceback
from dataclasses import asdict

from src.train.gpt import GPTTrainParams, GPTTrain
from src.utils.helper.connector import MultiProcessOutputConnector
from src.utils.response import EaseVoiceResponse, ResponseStatus


def main():
    connector = MultiProcessOutputConnector()
    try:
        parser = argparse.ArgumentParser(
            description="run train gpt"
        )
        parser.add_argument(
            "-c",
            "--config",
            type=argparse.FileType("r"),
            required=True,
        )
        args = parser.parse_args()
        config = json.loads(args.config)
        params = GPTTrainParams(**config)
        train = GPTTrain(params=params)
        output = train.train()
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.SUCCESS, message="FTraining GPT completed successfully", data=asdict(output)))
    except Exception as e:
        print(traceback.format_exc(), e)
        connector.write_response(EaseVoiceResponse(status=ResponseStatus.FAILED, message=f"{e}"))


if __name__ == "__main__":
    main()
