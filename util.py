import json
from dataclasses import asdict
import os
from typing import Any, Callable, List, Optional


class JSONObject:
    """
    dataclassをjsonファイルで保存, 読み込みができるようにするクラスです.

    使い方
    ```
    @dataclass
    class SomeDataClass(JSONObject):
        some_field1: str
        some_field2: int = field(default=0)

    data = SomeDataClass("some string")
    data.dump("data.json") # -> {"some_field1": "some string", "some_field2": 0}
    ```
    """

    @classmethod
    def load(
        cls, path: str, object_hook: Optional[Callable[[dict], Any]] = None
    ) -> "JSONObject":
        """
        jsonファイルのパスから読み込み, dataclassオブジェクトを生成します.

        Args:
            path (str): 読み込むjsonファイルのパス
            object_hook (Optional[Callable[[dict], Any]]): jsonファイルからオブジェクトを生成する関数

        Returns:
            JSONObject: 読み込んだdataclassオブジェクト
        """
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


def dir2paths(dir_path: str) -> List[str]:
    """
    ディレクトリに含まれるファイルパスのリストを返します.

    Args:
        dir_path (str): ディレクトリパス

    Returns:
        List[str]: ディレクトリに含まれるファイルパスのリスト
    """
    paths = list(map(lambda path: os.path.join(dir_path, path), os.listdir(dir_path)))

    return paths
