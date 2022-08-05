import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence

from deep_learning.util import JSONObject, dir2paths


class Dataset:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.__x = x
        self.__y = y

    def save(self, path: str):
        path = self.__validate_path(path)

        if os.path.exists(path + ".npz"):
            print(path + ".npz is already exists.")
            return

        Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
        np.savez(path, x=self.__x, y=self.__y)

    def split(self, split_rate: float) -> Tuple["Dataset", "Dataset"]:
        if split_rate <= 0 or split_rate >= 1:
            raise RuntimeError("0 < split_rate < 1")

        length = self.__x.shape[0]
        split_point = int(length * split_rate)

        return Dataset(self.__x[:split_point], self.__y[:split_point]), Dataset(
            self.__x[split_point:], self.__y[split_point:]
        )

    def normalize(self) -> None:
        self.__x -= np.mean(self.__x, axis=1)[:, np.newaxis]
        std = np.std(self.__x, axis=1)[:, np.newaxis]
        self.__x = np.divide(self.__x, std, out=np.zeros_like(self.__x), where=std != 0)

    def to_data_sequence(
        self, batch_size: int, batches_per_epoch: int = None, shuffle: bool = True
    ) -> "DataSequence":
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
        kcv_data_sequence = KCVDataSequence(
            self.__x, self.__y, k, batch_size, batches_per_epoch=batches_per_epoch
        )

        return kcv_data_sequence

    @classmethod
    def load(cls, path: str, shuffle=True) -> "Dataset":
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
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)

        np.random.seed(seed)
        np.random.shuffle(x)
        np.random.seed(seed)
        np.random.shuffle(y)


class DatasetConstructor:
    def __init__(
        self,
        data_paths: List[str],
        label_paths: List[str],
        process: Callable[[str, str], Tuple[np.ndarray, np.ndarray]],
    ) -> None:
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
        data_paths = sorted(dir2paths(data_dir))
        label_paths = sorted(dir2paths(label_dir))

        return cls(data_paths, label_paths, process)

    def construct(self, normalize=False, **kwargs) -> Dataset:
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
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        batches_per_epoch: int = None,
        shuffle: bool = True,
    ) -> None:
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
        return self.__batch_size

    @property
    def batches_per_epoch(self):
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
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        batch_size: int,
        batches_per_epoch: int = None,
    ) -> None:
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
        return self.__k

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def batches_per_epoch(self) -> int:
        return self.__batches_per_epoch

    def generate(self) -> Iterator[Tuple[int, DataSequence, DataSequence]]:
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
    batch_size: int
    epochs: int
    batches_per_epoch: Optional[int] = field(default=None)
