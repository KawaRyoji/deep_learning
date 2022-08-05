import json
from dataclasses import asdict
import os
from typing import Any, Callable


class JSONObject:
    @classmethod
    def load(cls, path: str, object_hook: Callable[[dict], Any] = None) -> "JSONObject":
        path = cls.__validate_path(path)

        with open(path, "r") as f:
            json_dict: dict = json.load(f, object_hook=object_hook)
            return cls(**json_dict)

    def dump(self, path: str, default: Callable[[Any], Any] = None) -> None:
        path = self.__validate_path(path)

        with open(path, "w") as f:
            json.dump(asdict(self), f, default=default)

    @staticmethod
    def __validate_path(path: str):
        return path if path.endswith(".json") else path + ".json"


def dir2paths(dir_path: str):
    paths = list(map(lambda path: os.path.join(dir_path, path), os.listdir(dir_path)))

    return paths
