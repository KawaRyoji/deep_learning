import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence

from deep_learning.util import JSONObject, dir2paths


class Dataset:
    """
    データセットを格納するクラスです.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            x (np.ndarray): データ
            y (np.ndarray): ラベル
        """
        self.__x = x
        self.__y = y

    def save(self, path: str):
        """
        データセットをnpzファイルに保存します.

        Args:
            path (str): 保存するファイルのパス
        """
        path = self.__validate_path(path)

        if os.path.exists(path + ".npz"):
            print(path + ".npz is already exists.")
            return

        Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
        np.savez(path, x=self.__x, y=self.__y)

    def split(self, split_rate: float) -> Tuple["Dataset", "Dataset"]:
        """
        割合によってデータセットを分割します.

        Args:
            split_rate (float): 分割する割合(0 < split_rate < 1)
        Raises:
            RuntimeError: split_rateが0以下または1以上の場合

        Returns:
            Dataset, Dataset: 分割したデータセット
        """
        if split_rate <= 0 or split_rate >= 1:
            raise RuntimeError("0 < split_rate < 1")

        length = self.__x.shape[0]
        split_point = int(length * split_rate)

        return Dataset(self.__x[:split_point], self.__y[:split_point]), Dataset(
            self.__x[split_point:], self.__y[split_point:]
        )

    def normalize(self) -> None:
        """
        データセットのデータを標準化します.
        """
        self.__x -= np.mean(self.__x, axis=1)[:, np.newaxis]
        std = np.std(self.__x, axis=1)[:, np.newaxis]
        self.__x = np.divide(self.__x, std, out=np.zeros_like(self.__x), where=std != 0)

    def to_data_sequence(
        self, batch_size: int, batches_per_epoch: int = None, shuffle: bool = True
    ) -> "DataSequence":
        """
        データセットからデータシークエンスを生成します.

        Args:
            batch_size (int): バッチサイズ
            batches_per_epoch (int, optional): エポック当たりのバッチ数
            shuffle (bool, optional): シャッフルするかどうか

        Returns:
            DataSequence: 生成したデータシークエンス
        """
        data_sequence = DataSequence(
            self.__x,
            self.__y,
            batch_size,
            batches_per_epoch=batches_per_epoch,
            shuffle=shuffle,
        )

        return data_sequence

    def to_kcv_data_sequence(
        self, batch_size: int, k: int, batches_per_epoch: int = None
    ) -> "KCVDataSequence":
        """
        データセットからk分割交差検証用データシークエンスを生成します.

        Args:
            batch_size (int): バッチサイズ
            k (int): 分割数
            batches_per_epoch (int, optional): エポック当たりのバッチ数

        Returns:
            KCVDataSequence: 生成したデータシークエンス
        """
        kcv_data_sequence = KCVDataSequence(
            self.__x, self.__y, k, batch_size, batches_per_epoch=batches_per_epoch
        )

        return kcv_data_sequence

    @classmethod
    def load(cls, path: str, shuffle=True) -> "Dataset":
        """
        データセットのパスからインスタンスを生成します.

        Args:
            path (str): 読み込むパス
            shuffle (bool, optional): シャッフルして読み込むかどうか

        Returns:
            Dataset: 生成したデータセットインスタンス
        """
        path = cls.__validate_path(path)

        data = np.load(path + ".npz", allow_pickle=True)
        x, y = data["x"], data["y"]
        del data  # コピーするのでメモリ確保に行う

        if shuffle:
            cls.shuffle(x, y)

        return cls(x, y)

    @staticmethod
    def __validate_path(path: str) -> str:
        if path.endswith(".npz"):
            return path[:-4]

        return path

    @staticmethod
    def shuffle(x: np.ndarray, y: np.ndarray, seed: int = None) -> None:
        """
        データセットをインメモリでシャッフルします.

        Args:
            x (np.ndarray): データ
            y (np.ndarray): ラベル
            seed (int, optional): シャッフルするときのシード値
        """
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)

        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)


class DatasetConstructor:
    """
    ファイルからデータセットを構築するクラスです.
    """

    def __init__(
        self,
        data_paths: List[str],
        label_paths: List[str],
        process: Callable[[str, str], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """
        Args:
            data_paths (List[str]): データのパスのリスト
            label_paths (List[str]): ラベルのパスのリスト
            process (Callable[[str, str], Tuple[np.ndarray, np.ndarray]]): データのパスとラベルのパスでデータを生成する関数

        Example:
        ```
        def some_process(data_path, label_path):
            data = read_data(data_path)
            label = read_label(label_path)
            # ここで読み込んだデータに対して処理
            return data, label
        ```
        """
        self.__data_paths = np.array(data_paths)
        self.__label_paths = np.array(label_paths)
        self.process = process

    @classmethod
    def from_dir(
        cls,
        data_dir: str,
        label_dir: str,
        process: Callable[[str, str], Tuple[np.ndarray, np.ndarray]],
    ) -> "DatasetConstructor":
        """
        ディレクトリパスからインスタンスを生成します.

        Args:
            data_dir (str): データのディレクトリパス
            label_dir (str): ラベルのディレクトリパス
            process (Callable[[str, str], Tuple[np.ndarray, np.ndarray]]): データのパスとラベルのパスでデータを生成する関数

        Returns:
            DatasetConstructor: _description_
        """
        data_paths = sorted(dir2paths(data_dir))
        label_paths = sorted(dir2paths(label_dir))

        return cls(data_paths, label_paths, process)

    def construct(self, normalize=False, **kwargs) -> Dataset:
        """
        設定したデータセットを生成する関数でデータセットを生成します.

        Args:
            normalize (bool, optional): 標準化をするかどうか

        Returns:
            Dataset: 生成したデータセット
        """
        data_list: List[np.ndarray] = []
        label_list: List[np.ndarray] = []
        for data_path, label_path in zip(self.__data_paths, self.__label_paths):
            data, label = self.process(data_path, label_path, **kwargs)
            data_list.extend(data)
            label_list.extend(label)

        data_np: np.ndarray = np.array(data_list, dtype=np.float32)
        label_np: np.ndarray = np.array(label_list, dtype=np.float32)

        dataset = Dataset(data_np, label_np)

        if normalize:
            dataset.normalize()

        return dataset


class DataSequence(Sequence):
    """
    model.fit()に渡すデータシークエンスのクラスです.
    このクラスでバッチサイズやエポックあたりのバッチ数を調整できます.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        batches_per_epoch: int = None,
        shuffle: bool = True,
    ) -> None:
        """
        Args:
            x (np.ndarray): データ
            y (np.ndarray): ラベル
            batch_size (int): バッチサイズ
            batches_per_epoch (int, optional): エポック当たりのバッチ数
            shuffle (bool, optional): エポックごとにシャッフルするかどうか
        """
        self.__x = x
        self.__y = y
        self.__batch_size = batch_size
        if batches_per_epoch is None:
            self.__batches_per_epoch: int = x.shape[0] // batch_size
        else:
            self.__batches_per_epoch = batches_per_epoch
        self.__shuffle = shuffle

    @property
    def batch_size(self):
        """
        バッチサイズ
        """
        return self.__batch_size

    @property
    def batches_per_epoch(self):
        """
        エポックあたりのバッチ数
        """
        return self.__batches_per_epoch

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size

        return self.__x[start:end], self.__y[start:end]

    def on_epoch_end(self):
        if self.__shuffle:
            Dataset.shuffle(self.__x, self.__y)


class KCVDataSequence:
    """
    k分割交差検証で用いるデータシークエンスを生成するクラスです.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        batch_size: int,
        batches_per_epoch: int = None,
    ) -> None:
        """
        Args:
            x (np.ndarray): データ
            y (np.ndarray): ラベル
            k (int): 分割数
            batch_size (int): バッチサイズ
            batches_per_epoch (int, optional): エポック当たりのバッチ数
        """
        self.__x = x
        self.__y = y
        self.__k = k
        self.__batch_size = batch_size

        if batches_per_epoch is None:
            self.__batches_per_epoch: int = x.shape[0] // self.__batch_size
        else:
            self.__batches_per_epoch = batches_per_epoch

    @property
    def k(self) -> int:
        """
        分割数
        """
        return self.__k

    @property
    def batch_size(self) -> int:
        """
        バッチサイズ
        """
        return self.__batch_size

    @property
    def batches_per_epoch(self) -> int:
        """
        エポック当たりのバッチ数
        """
        return self.__batches_per_epoch

    def generate(self) -> Iterator[Tuple[int, DataSequence, DataSequence]]:
        """
        各foldに使う学習シークエンスを生成する関数です.

        Yields:
            Iterator[Tuple[int, DataSequence, DataSequence]]: fold数, 学習シークエンス, 検証シークエンス
        """
        fold_size = self.__x.shape[0] // self.k

        for fold in range(self.k):
            start = fold * fold_size
            end = start + fold_size

            mask = np.ones(self.__x.shape[0], dtype=bool)
            mask[start:end] = False

            x_valid = self.__x[start:end]
            y_valid = self.__y[start:end]

            x_train = self.__x[mask]
            y_train = self.__y[mask]

            del mask
            train_sequence = DataSequence(
                x_train,
                y_train,
                self.batch_size,
                batches_per_epoch=int(self.batches_per_epoch * (self.k - 1) / self.k),
            )

            valid_sequence = DataSequence(
                x_valid,
                y_valid,
                self.batch_size,
                batches_per_epoch=int(self.batches_per_epoch / self.k),
                shuffle=False,
            )

            yield (fold, train_sequence, valid_sequence)


@dataclass(frozen=True)
class DatasetParams(JSONObject):
    """
    データセットに用いるパラメータのクラスです.
    """

    batch_size: int
    epochs: int
    batches_per_epoch: Optional[int] = field(default=None)
