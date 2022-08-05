import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint

from prototype.dataset import Dataset, DatasetParams
from prototype.dnn import DNN, CheckPoint, CheckPointCallBack, LearningHistory


@dataclass(frozen=True)
class HoldoutDirectory:
    root_dir: str
    holdout_dir: str = field(init=False, compare=False)
    figures_dir: str = field(init=False, compare=False)
    history_path: str = field(init=False, compare=False)
    checkpoint_path: str = field(init=False, compare=False)
    model_weight_dir: str = field(init=False, compare=False)
    best_weight_path: str = field(init=False, compare=False)
    latest_weight_path: str = field(init=False, compare=False)
    test_result_path: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        setattr(
            self,
            "holdout_dir",
            os.path.join(self.root_dir, "holdout"),
        )
        setattr(
            self,
            "figures_dir",
            os.path.join(self.holdout_dir, "figures"),
        )
        setattr(
            self,
            "history_path",
            os.path.join(self.holdout_dir, "history.csv"),
        )
        setattr(
            self, "test_result_path", os.path.join(self.holdout_dir, "test_result.csv")
        )
        setattr(
            self,
            "checkpoint_path",
            os.path.join(self.holdout_dir, "check_point.json"),
        )
        setattr(
            self,
            "model_weight_dir",
            os.path.join(self.holdout_dir, "model_weights"),
        )
        setattr(
            self,
            "best_weight_path",
            os.path.join(self.model_weight_dir, "best_model.ckpt"),
        )
        setattr(
            self,
            "latest_weight_path",
            os.path.join(self.model_weight_dir, "latest_model.ckpt"),
        )

        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.holdout_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_weight_dir).mkdir(parents=True, exist_ok=True)

    def figure_path(self, file_path: str) -> str:
        return os.path.join(self.figures_dir, file_path)

    def model_weight_path(self, weight_path: str) -> str:
        return os.path.join(self.model_weight_dir, weight_path)


@dataclass(frozen=True)
class KCVDirectory:
    root_dir: str
    kcv_dir: str = field(init=False, compare=False)
    figures_dir: str = field(init=False, compare=False)
    figure_average: str = field(init=False, compare=False)
    histories_dir: str = field(init=False, compare=False)
    checkpoint_path: str = field(init=False, compare=False)
    model_weight_dir: str = field(init=False, compare=False)
    latest_weight_path: str = field(init=False, compare=False)
    test_result_path: str = field(init=False, compare=False)

    def __post_init__(self):
        setattr(
            self,
            "kcv_dir",
            os.path.join(self.root_dir, "kcv"),
        )
        setattr(
            self,
            "figures_dir",
            os.path.join(self.kcv_dir, "figures"),
        )
        setattr(
            self,
            "model_weight_dir",
            os.path.join(self.kcv_dir, "model_weights"),
        )
        setattr(
            self,
            "histories_dir",
            os.path.join(self.kcv_dir, "histories"),
        )
        setattr(
            self,
            "test_result_path",
            os.path.join(self.kcv_dir, "test_result.csv"),
        )
        setattr(
            self,
            "latest_weight_path",
            os.path.join(self.model_weight_dir, "latest_model.ckpt"),
        )
        setattr(
            self,
            "figure_average",
            os.path.join(self.figures_dir, "average"),
        )

        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.kcv_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figure_average).mkdir(parents=True, exist_ok=True)
        Path(self.histories_dir).mkdir(parents=True, exist_ok=True)

    def figure_fold_dir(self, fold: int) -> str:
        path = os.path.join(self.figures_dir, "fold%d" % fold)

        Path(path).mkdir(parents=True, exist_ok=True)

        return path

    def history_path(self, fold: int) -> str:
        return os.path.join(self.histories_dir, "history_fold%d.csv" % fold)

    def best_weight_path(self, fold: int) -> str:
        return os.path.join(
            self.model_weight_dir, "best_model_weight_fold%d.ckpt" % fold
        )


class DNNExperiment:
    def __init__(
        self,
        dnn: DNN,
        root_dir: str,
        train_set: Dataset,
        test_set: Dataset,
        dataset_params: DatasetParams,
        model_params: dict = None,
        train_method: str = "holdout",
        k: Optional[int] = None,
        valid_split: Optional[float] = None,
        gpu: Optional[int] = None,
    ) -> None:
        self.__set_hardware(gpu_i=gpu)

        if train_method == "holdout":
            self.__directory = HoldoutDirectory(root_dir)
        elif train_method == "kcv":
            if k is None:
                raise RuntimeError("k must not be None.")
            self.__directory = KCVDirectory(root_dir)
            self.__k = k
        else:
            raise RuntimeError("Train method assumes 'holdout' or 'kcv'.")

        self.__dnn = dnn
        self.__train_method = train_method
        self.__model_params = model_params
        self.__train_set = train_set
        self.__test_set = test_set
        self.__dataset_params = dataset_params
        self.__valid_split = valid_split

    @classmethod
    def from_dataset_path(
        cls,
        dnn: DNN,
        root_dir: str,
        train_set_path: str,
        test_set_path: str,
        dataset_params: DatasetParams,
        model_params: dict = None,
        train_method: str = "holdout",
        k: Optional[int] = None,
        valid_split: Optional[float] = None,
        gpu: Optional[int] = None,
    ) -> "DNNExperiment":
        train_set = Dataset.load(train_set_path)
        test_set = Dataset.load(test_set_path, shuffle=False)

        return cls(
            dnn=dnn,
            root_dir=root_dir,
            train_set=train_set,
            test_set=test_set,
            dataset_params=dataset_params,
            model_params=model_params,
            train_method=train_method,
            k=k,
            valid_split=valid_split,
            gpu=gpu,
        )

    def __set_hardware(self, gpu_i: Optional[int] = None) -> None:
        if gpu_i is None:
            print("Experiment will run on CPU.")
            return

        try:
            physical_devices = tf.config.list_physical_devices("GPU")
            tf.config.set_visible_devices(physical_devices[gpu_i], "GPU")
            tf.config.experimental.set_memory_growth(physical_devices[gpu_i], True)
            print("Experiment will run on GPU{}.".format(gpu_i))
        except:
            print("Can't use GPU{}. Experiment will run on CPU.".format(gpu_i))

    def train(
        self,
        *args,
        check_point: Optional[CheckPoint] = None,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        clear_session()  # メモリリーク対策
        if self.__train_method == "holdout":
            self.__holdout_train(
                *args,
                check_point=check_point,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                callbacks=additional_callbacks,
            )
        elif self.__train_method == "kcv":
            self.__kcv_train(
                *args,
                check_point=check_point,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                callbacks=additional_callbacks,
            )

    def __holdout_train(
        self,
        *args,
        check_point: Optional[CheckPoint] = None,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.__dnn.compile(*args, **self.__model_params)

        callbacks = [
            CSVLogger(self.__directory.history_path),
            CheckPointCallBack(
                self.__directory.checkpoint_path,
                self.__directory.latest_weight_path,
            ),
            ModelCheckpoint(
                self.__directory.best_weight_path(),
                monitor=monitor_metric,
                mode=monitor_mode,
            ),
        ]

        if additional_callbacks is not None:
            callbacks.extend(additional_callbacks)

        if self.__valid_split is None:
            train_sequence = self.__train_set.to_data_sequence(
                batch_size=self.__dataset_params.batch_size,
                batches_per_epoch=self.__dataset_params.batches_per_epoch,
            )

            valid_sequence = None
        else:
            train_set, valid_set = self.__train_set.split(self.__valid_split)

            train_sequence = train_set.to_data_sequence(
                batch_size=self.__dataset_params.batch_size,
                batches_per_epoch=self.__dataset_params.batches_per_epoch,
            )

            valid_sequence = valid_set.to_data_sequence(
                batch_size=self.__dataset_params.batch_size,
                batches_per_epoch=self.__dataset_params.batches_per_epoch,
            )

        self.__dnn.train(
            train_sequence=train_sequence,
            valid_sequence=valid_sequence,
            epochs=self.__dataset_params.epochs,
            check_point=check_point,
            callbacks=callbacks,
        )

    def __kcv_train(
        self,
        *args,
        check_point: Optional[CheckPoint] = None,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.__dnn.compile(*args, **self.__model_params)

        sequence = self.__train_set.to_kcv_data_sequence(
            self.__dataset_params.batch_size,
            self.__k,
            batches_per_epoch=self.__dataset_params.batches_per_epoch,
        )

        if check_point is not None:
            init_fold = 0
        else:
            init_fold = check_point.fold

        for fold, train_sequence, valid_sequence in sequence.generate():
            if init_fold > fold:
                continue
            clear_session()  # メモリリーク対策

            callbacks = [
                CSVLogger(self.__directory.history_path(fold)),
                CheckPointCallBack(
                    self.__directory.checkpoint_path,
                    self.__directory.latest_weight_path,
                    fold=fold,
                ),
                ModelCheckpoint(
                    self.__directory.best_weight_path(fold),
                    monitor=monitor_metric,
                    mode=monitor_mode,
                ),
            ]

            if additional_callbacks is not None:
                callbacks.extend(additional_callbacks)

            self.__dnn.train(
                train_sequence=train_sequence,
                valid_sequence=valid_sequence,
                epochs=self.__dataset_params.epochs,
                check_point=check_point,
                callbacks=callbacks,
            )

    def test(self) -> None:
        if self.__train_method == "holdout":
            self.__holdout_test()
        elif self.__train_method == "kcv":
            self.__kcv_test()

    def __holdout_test(self) -> None:
        test_sequence = self.__test_set.to_data_sequence(1, shuffle=False)

        test_result = self.__dnn.test(
            test_sequence,
            model_weight_path=self.__directory.best_weight_path,
        )

        history = LearningHistory(test_result, self.__dnn.get_metrics())
        history.save_to(self.__directory.test_result_path)

    def __kcv_test(self) -> None:
        test_sequence = self.__test_set.to_data_sequence(1, shuffle=False)

        results = []
        for fold in range(self.__k):
            test_result = self.__dnn.test(
                test_sequence,
                model_weight_path=self.__directory.best_weight_path(fold),
            )

            results.append(test_result)

        df = pd.DataFrame(
            results,
            index=[i for i in range(self.__k)],
            columns=["fold"] + self.__dnn.get_metrics(),
        )

        df.to_csv(self.__directory.test_result_path)
