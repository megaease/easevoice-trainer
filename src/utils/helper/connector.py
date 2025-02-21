import json
import select
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.response import EaseVoiceResponse


class ConnectorDataType:
    RESP = "response"
    LOSS = "loss"
    LOG = "LOG"
    OTHER = "other"
    SESSION_DATA = "session_data"


@dataclass
class ConnectorDataLoss:
    step: int
    loss: float
    other: dict


@dataclass
class ConnectorData:
    dataType: str
    response: Optional[EaseVoiceResponse] = None
    loss: Optional[ConnectorDataLoss] = None
    log: Optional[dict] = None
    other: Optional[str] = None
    session_data: Optional[dict] = None


class MultiProcessOutputConnector:
    """
    MultiProcessOutputConnector: works like multiprocessing Queue.
    When use Popen start a new process, write result or log use this class.
    Then main process can easily read data from stdout or stderr use this class.
    """

    def __init__(self):
        self._resp_prefix = "response-of-easevoice"
        self._loss_prefix = "loss-of-easevoice"
        self._log_prefix = "log-of-easevoice"
        self._session_data_prefix = "session-data-of-easevoice"

    def _print(self, prefix: str, data: str):
        print(f"{prefix} {data}", flush=True)

    def write_response(self, resp: EaseVoiceResponse):
        data = json.dumps(resp.to_dict())  # pyright: ignore
        self._print(self._resp_prefix, data)

    def write_session_data(self, data: dict):
        json_data = json.dumps(data)
        self._print(self._session_data_prefix, json_data)

    def write_loss(self, step: int, loss: Any, other: Optional[dict] = None):
        data = {
            "step": step,
            "loss": loss
        }
        if other is not None:
            data.update(other)
        json_data = json.dumps(data)
        self._print(self._loss_prefix, json_data)

    def write_log(self, log: dict):
        json_data = json.dumps(log)
        self._print(self._log_prefix, json_data)

    def read_data(self, process: subprocess.Popen):
        while True:
            ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

            for stream in ready:
                line = stream.readline()
                if not line:
                    continue

                if isinstance(line, bytes):
                    line = line.decode('utf-8')

                parsed = self._parse_result(line.strip())
                if parsed is not None:
                    if parsed.dataType == ConnectorDataType.OTHER:
                        print(parsed.other)
                    else:
                        yield parsed

            if process.poll() is not None:
                for stream in [process.stdout, process.stderr]:
                    if stream:
                        try:
                            remaining = stream.read()
                            if remaining:
                                if isinstance(remaining, bytes):
                                    remaining = remaining.decode('utf-8')
                                for l in remaining.splitlines():
                                    parsed = self._parse_result(l.strip())
                                    if parsed is not None:
                                        if parsed.dataType == ConnectorDataType.OTHER:
                                            print(parsed.other)
                                        else:
                                            yield parsed
                        except ValueError:
                            print("Error when reading remaining output")

                break

        process.wait()

    def _parse_result(self, line: str):
        try:
            if line.startswith(self._resp_prefix):
                data = line[len(self._resp_prefix):].strip()
                data = json.loads(data)
                return ConnectorData(dataType=ConnectorDataType.RESP, response=EaseVoiceResponse(**data))

            elif line.startswith(self._loss_prefix):
                data = line[len(self._loss_prefix):].strip()
                data = json.loads(data)
                loss = data["loss"]
                step = data["step"]
                data.pop("loss")
                data.pop("step")
                l = ConnectorDataLoss(step, loss, data)
                return ConnectorData(dataType=ConnectorDataType.LOSS, loss=l)

            elif line.startswith(self._log_prefix):
                data = line[len(self._log_prefix):].strip()
                data = json.loads(data)
                return ConnectorData(dataType=ConnectorDataType.LOG, log=data)

            elif line.startswith(self._session_data_prefix):
                data = line[len(self._session_data_prefix):].strip()
                data = json.loads(data)
                return ConnectorData(dataType=ConnectorDataType.SESSION_DATA, session_data=data)
            else:
                return ConnectorData(dataType=ConnectorDataType.OTHER, other=line)
        except Exception as e:
            print(f"meet error when parse stdout: {e}, input: <{line}>")
        return None
