import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint

from deep_learning.dataset import Dataset, DatasetParams
from deep_learning.dnn import DNN, CheckPoint, CheckPointCallBack, LearningHistory
from deep_learning.plot import HistoryPlotter


@dataclass
class HoldoutDirectory:
    """
    ホールドアウト法で結果を保存する際の, ディレクトリパスとファイルパスを格納するクラスです.
    """

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
        """
        root_dirに従って, フィールドのパスを設定します.
        """
        self.holdout_dir = os.path.join(self.root_dir, "holdout")

        self.figures_dir = os.path.join(self.holdout_dir, "figures")
        self.history_path = os.path.join(self.holdout_dir, "history.csv")
        self.test_result_path = os.path.join(self.holdout_dir, "test_result.csv")
        self.checkpoint_path = os.path.join(self.holdout_dir, "check_point.json")
        self.model_weight_dir = os.path.join(self.holdout_dir, "model_weights")

        self.best_weight_path = os.path.join(self.model_weight_dir, "best_model.ckpt")
        self.latest_weight_path = os.path.join(
            self.model_weight_dir, "latest_model.ckpt"
        )

        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.holdout_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_weight_dir).mkdir(parents=True, exist_ok=True)

    def figure_path(self, file_path: str) -> str:
        """
        画像のパスを画像ディレクトリパスと結合して返します.

        Args:
            file_path (str): 画像のパス

        Returns:
            str: 画像のパス
        """
        return os.path.join(self.figures_dir, file_path)

    def model_weight_path(self, weight_path: str) -> str:
        return os.path.join(self.model_weight_dir, weight_path)


@dataclass
class KCVDirectory:
    """
    k分割交差検証で結果を保存する際の, ディレクトリパスとファイルパスを格納するクラスです.
    """

    root_dir: str
    kcv_dir: str = field(init=False, compare=False)
    figures_dir: str = field(init=False, compare=False)
    figure_average: str = field(init=False, compare=False)
    histories_dir: str = field(init=False, compare=False)
    checkpoint_path: str = field(init=False, compare=False)
    model_weight_dir: str = field(init=False, compare=False)
    latest_weight_path: str = field(init=False, compare=False)
    test_result_path: str = field(init=False, compare=False)
    test_result_figure_path: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """
        root_dirに従って, フィールドのパスを設定します.
        """
        self.kcv_dir = os.path.join(self.root_dir, "kcv")

        self.figures_dir = os.path.join(self.kcv_dir, "figures")
        self.model_weight_dir = os.path.join(self.kcv_dir, "model_weights")
        self.histories_dir = os.path.join(self.kcv_dir, "histories")
        self.test_result_path = os.path.join(self.kcv_dir, "test_result.csv")
        self.checkpoint_path = os.path.join(self.kcv_dir, "check_point.json")

        self.latest_weight_path = os.path.join(
            self.model_weight_dir, "latest_model.ckpt"
        )

        self.figure_average = os.path.join(self.figures_dir, "average")
        self.test_result_figure_path = os.path.join(self.figures_dir, "test_result.png")

        Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        Path(self.kcv_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figure_average).mkdir(parents=True, exist_ok=True)
        Path(self.histories_dir).mkdir(parents=True, exist_ok=True)

    def figure_fold_dir(self, fold: int) -> str:
        """
        foldにおけるディレクトリパスを画像ディレクトリパスと結合して返します.

        Args:
            fold (int): ディレクトリのfold

        Returns:
            str: 画像ディレクトリパス
        """
        path = os.path.join(self.figures_dir, "fold%d" % fold)

        Path(path).mkdir(parents=True, exist_ok=True)

        return path

    def history_path(self, fold: int) -> str:
        """
        学習履歴のパスをを学習履歴ディレクトリパスと結合して返します.

        Args:
            fold (int): パスのfold

        Returns:
            str: 学習履歴のパス
        """
        return os.path.join(self.histories_dir, "history_fold%d.csv" % fold)

    def best_weight_path(self, fold: int) -> str:
        """
        各foldの一番性能のよいモデルの重みを保存するパスを返します.

        Args:
            fold (int): パスのfold

        Returns:
            str: 重みのパス
        """
        return os.path.join(
            self.model_weight_dir, "best_model_weight_fold%d.ckpt" % fold
        )


class HoldoutExperiment:
    """
    ホールドアウト法での実験を行うクラスです.
    """

    def __init__(
        self,
        dnn: DNN,
        root_dir: str,
        train_set: Dataset,
        test_set: Dataset,
        dataset_params: DatasetParams,
        valid_split: Optional[float] = None,
        gpu: Optional[int] = None,
    ) -> None:
        """
        Args:
            dnn (DNN): DNNモデル
            root_dir (str): 結果保存先のディレクトリパス
            train_set (Dataset): 学習用データセット
            test_set (Dataset): テスト用データセット
            dataset_params (DatasetParams): データセットのパラメータ
            valid_split (Optional[int], optional): 検証データの割合
            gpu (Optional[int], optional): 使用するGPU番号
        """
        self.__set_hardware(gpu_i=gpu)

        self.__dnn = dnn
        self.__directory = HoldoutDirectory(root_dir)
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
        valid_split: Optional[float] = None,
        gpu: Optional[int] = None,
    ) -> "HoldoutExperiment":
        """
        datasetの保存先パスからインスタンスを生成します.

        Args:
            dnn (DNN): DNNモデル
            root_dir (str): 結果保存先のディレクトリパス
            train_set_path (str): 学習用データセットのパス
            test_set_path (str): テスト用データセットのパス
            dataset_params (DatasetParams):  データセットのパラメータ
            model_params (dict, optional):  モデルのパラメータ
            valid_split (Optional[float], optional):  検証用データの割合
            gpu (Optional[int], optional):  使用するGPU番号

        Returns:
            KCVExperiment: 生成したインスタンス
        """
        train_set = Dataset.load(train_set_path)
        test_set = Dataset.load(test_set_path, shuffle=False)

        return cls(
            dnn=dnn,
            root_dir=root_dir,
            train_set=train_set,
            test_set=test_set,
            dataset_params=dataset_params,
            model_params=model_params,
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
        use_checkpoint: bool = True,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """
        モデルを学習します.

        Args:
            use_checkpoint (bool, optional): チェックポイントから学習を再開させるかどうか
            monitor_metric (str, optional): モデルの評価値　この評価値で性能の良い重みを決定します
            monitor_mode (str, optional): モデルの評価値の監視モード
            additional_callbacks (Optional[List[Callback]], optional): 追加するコールバック関数
        """
        self.__dnn.compile()
        directory: HoldoutDirectory = self.__directory

        callbacks = [
            CSVLogger(directory.history_path),
            CheckPointCallBack(
                directory.checkpoint_path,
                directory.latest_weight_path,
            ),
            ModelCheckpoint(
                directory.best_weight_path,
                monitor=monitor_metric,
                mode=monitor_mode,
            ),
        ]

        if additional_callbacks is not None:
            callbacks.extend(additional_callbacks)

        if self.__valid_split is None:
            train_set, valid_set = self.__train_set, None
        else:
            train_set, valid_set = self.__train_set.split(self.__valid_split)

        checkpoint = self.__load_checkpoint(
            use_checkpoint, self.__dataset_params.epochs
        )

        self.__dnn.train(
            train_set=train_set,
            valid_set=valid_set,
            epochs=self.__dataset_params.epochs,
            batch_size=self.__dataset_params.batch_size,
            steps_per_epoch=self.__dataset_params.steps_per_epoch,
            checkpoint=checkpoint,
            callbacks=callbacks,
        )

        self.__used_checkpoint(directory.history_path)

    def __load_checkpoint(
        self, use_checkpoint: bool, epochs: int
    ) -> Union[CheckPoint, None]:
        if not use_checkpoint:
            return None

        checkpoint = CheckPoint.load(self.__directory.checkpoint_path)
        if checkpoint is None:
            return None

        if checkpoint.epoch == epochs - 1:  # チェックポイントが最終エポックの場合
            return checkpoint

        tmp_history = LearningHistory.from_path(self.__directory.history_path)
        tmp_history.save_to(os.path.join(self.__directory.holdout_dir, "tmp.csv"))

        return checkpoint

    def __used_checkpoint(self, new_history_path: str) -> None:
        if os.path.exists(os.path.join(self.__directory.holdout_dir, "tmp.csv")):
            old_history = LearningHistory.from_path(
                os.path.join(self.__directory.holdout_dir, "tmp.csv")
            )
            new_history = LearningHistory.from_path(new_history_path)
            history = LearningHistory.concat([old_history, new_history])
            history.save_to(new_history_path)
            os.remove(os.path.join(self.__directory.holdout_dir, "tmp.csv"))

    def test(self) -> None:
        """
        モデルでテストを行います.
        すでにテスト結果がある場合はスキップされます.
        """
        if os.path.exists(self.__directory.test_result_path):
            return

        self.__dnn.compile()

        test_result = self.__dnn.test(
            self.__test_set,
            model_weight_path=self.__directory.best_weight_path,
        )

        history = LearningHistory.from_list([test_result], self.__metrics_without_val())
        history.save_to(self.__directory.test_result_path)

    def __metrics_without_val(self) -> List[str]:
        return list(
            filter(
                lambda name: not name.startswith("val_") and not name == "epoch",
                self.__dnn.get_metrics(),
            )
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        一番性能の良い重みで推論を行います.

        Args:
            x (np.ndarray): 推論に使うデータ

        Returns:
            np.ndarray: 推論の結果
        """

        self.__dnn.compile()

        prediction = self.__dnn.predict(
            x, model_weight_path=self.__directory.best_weight_path
        )

        return prediction

    def plot(self) -> None:
        """
        学習履歴から結果をプロットします.
        """
        HistoryPlotter.set_style()
        history = LearningHistory.from_path(self.__directory.history_path)

        plotter = HistoryPlotter(history)
        plotter.plot_all_metrics(self.__directory.figures_dir)


class KCVExperiment:
    """
    k分割交差検証での実験を行うクラスです.
    """

    def __init__(
        self,
        dnn: DNN,
        root_dir: str,
        k: int,
        train_set: Dataset,
        test_set: Dataset,
        dataset_params: DatasetParams,
        gpu: Optional[int] = None,
    ) -> None:
        """
        Args:
            dnn (DNN): DNNモデル
            root_dir (str): 結果保存先のディレクトリパス
            k (int): k分割交差検証のパラメータ
            train_set (Dataset): 学習用データセット
            test_set (Dataset): テスト用データセット
            dataset_params (DatasetParams): データセットのパラメータ
            gpu (Optional[int], optional): 使用するGPU番号
        """
        self.__set_hardware(gpu_i=gpu)

        self.__dnn = dnn
        self.__directory = KCVDirectory(root_dir)
        self.__k = k
        self.__train_set = train_set
        self.__test_set = test_set
        self.__dataset_params = dataset_params

    @classmethod
    def from_dataset_path(
        cls,
        dnn: DNN,
        root_dir: str,
        k: int,
        train_set_path: str,
        test_set_path: str,
        dataset_params: DatasetParams,
        model_params: dict = None,
        gpu: Optional[int] = None,
    ) -> "KCVExperiment":
        """
        datasetの保存先パスからインスタンスを生成します.

        Args:
            dnn (DNN): DNNモデル
            root_dir (str): 結果保存先のディレクトリパス
            k (int):  k分割交差検証のパラメータ
            train_set_path (str): 学習用データセットのパス
            test_set_path (str): テスト用データセットのパス
            dataset_params (DatasetParams):  データセットのパラメータ
            train_method (str, optional):  学習方法("holdout" または "kcv")
            gpu (Optional[int], optional):  使用するGPU番号

        Returns:
            KCVExperiment: 生成したインスタンス
        """
        train_set = Dataset.load(train_set_path)
        test_set = Dataset.load(test_set_path, shuffle=False)

        return cls(
            dnn=dnn,
            root_dir=root_dir,
            train_set=train_set,
            test_set=test_set,
            dataset_params=dataset_params,
            model_params=model_params,
            k=k,
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
        except Exception as e:
            print("Can't use GPU{}. Experiment will run on CPU.".format(gpu_i))
            print(e)

    def train(
        self,
        use_checkpoint: bool = True,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """
        モデルを学習します.

        Args:
            use_checkpoint (bool, optional): チェックポイントから学習を再開させるかどうか
            monitor_metric (str, optional): モデルの評価値　この評価値で性能の良い重みを決定します
            monitor_mode (str, optional): モデルの評価値の監視モード
            additional_callbacks (Optional[List[Callback]], optional): 追加するコールバック関数
        """
        data_len = self.__train_set.x.shape[0]
        fold_size = data_len // self.__k
        batch_size = self.__dataset_params.batch_size

        if self.__dataset_params.steps_per_epoch is None:
            steps_per_epoch = data_len // self.__dataset_params.batch_size
        else:
            steps_per_epoch = self.__dataset_params.steps_per_epoch

        checkpoint = self.__load_checkpoint(
            use_checkpoint, self.__dataset_params.epochs
        )

        if checkpoint is None:
            init_fold = 0
        else:
            init_fold = checkpoint.fold

        for fold in range(self.__k):
            if init_fold > fold:  # チェックポイントに記述されたfoldまで飛ばす
                continue

            start = fold * fold_size
            end = start + fold_size

            train_set, valid_set = self.__train_set.cv_mask(start, end)

            def train_gen() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
                indexes = np.arange(train_set.x.shape[0])
                while True:
                    sample_indexes = np.random.choice(indexes, size=batch_size)
                    yield (train_set.x[sample_indexes], train_set.y[sample_indexes])

            def valid_gen() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
                for i in range(0, valid_set.x.shape[0] // batch_size):
                    start, end = i * batch_size, (i + 1) * batch_size
                    yield (valid_set.x[start:end], valid_set.y[start:end])

            tf_train_set = tf.data.Dataset.from_generator(
                train_gen,
                output_types=(train_set.x.dtype, train_set.y.dtype),
                output_shapes=(
                    (batch_size, *train_set.x.shape[1:]),
                    (batch_size, *train_set.y.shape[1:]),
                ),
            ).repeat()
            tf_valid_set = tf.data.Dataset.from_generator(
                valid_gen,
                output_types=(valid_set.x.dtype, valid_set.y.dtype),
                output_shapes=(
                    (batch_size, *valid_set.x.shape[1:]),
                    (batch_size, *valid_set.y.shape[1:]),
                ),
            ).repeat()

            # valid_x = valid_set.x[:steps_per_epoch // self.__k]
            # valid_y = valid_set.y[:steps_per_epoch // self.__k]

            callbacks = [
                CSVLogger(self.__directory.history_path(fold)),
                # CheckPointCallBack(
                #     self.__directory.checkpoint_path,
                #     self.__directory.latest_weight_path,
                #     fold=fold,
                # ),
                ModelCheckpoint(
                    self.__directory.best_weight_path(fold),
                    monitor=monitor_metric,
                    mode=monitor_mode,
                ),
            ]

            if additional_callbacks is not None:
                callbacks.extend(additional_callbacks)

            self.__dnn.compile()

            self.__dnn.train(
                train_dataset=tf_train_set,
                valid_dataset=tf_valid_set,
                steps_per_epoch=(steps_per_epoch * (self.__k - 1)) // self.__k,
                validation_steps=steps_per_epoch // self.__k,
                epochs=self.__dataset_params.epochs,
                checkpoint=checkpoint,
                callbacks=callbacks,
            )
            del train_set, valid_set, tf_train_set, tf_valid_set
            gc.collect()

            # チェックポイントから始めた場合に元の学習履歴が消去されてしまう問題を解決する
            self.__used_checkpoint(self.__directory.history_path(fold))
            checkpoint = None  # 一度チェックポイントを使ったら使わない

    def __used_checkpoint(self, new_history_path: str) -> None:
        if os.path.exists(os.path.join(self.__directory.histories_dir, "tmp.csv")):
            old_history = LearningHistory.from_path(
                os.path.join(self.__directory.histories_dir, "tmp.csv")
            )
            new_history = LearningHistory.from_path(new_history_path)
            history = LearningHistory.concat([old_history, new_history])
            history.save_to(new_history_path)
            os.remove(os.path.join(self.__directory.histories_dir, "tmp.csv"))

    def __load_checkpoint(
        self, use_checkpoint: bool, epochs: int
    ) -> Union[CheckPoint, None]:
        if not use_checkpoint:
            return None

        checkpoint = CheckPoint.load(self.__directory.checkpoint_path)
        if checkpoint is None:
            return None

        if checkpoint.fold is None:
            raise RuntimeError("fold of checkpoint must not be None.")

        if checkpoint.epoch == epochs - 1:  # チェックポイントが最終エポックの場合
            return checkpoint

        tmp_history = LearningHistory.from_path(
            self.__directory.history_path(checkpoint.fold)
        )
        tmp_history.save_to(os.path.join(self.__directory.histories_dir, "tmp.csv"))

        return checkpoint

    def test(self) -> None:
        """
        モデルでテストを行います.
        すでにテスト結果がある場合はスキップされます.
        """
        if os.path.exists(self.__directory.test_result_path):
            return

        self.__dnn.compile()

        results = []
        for fold in range(self.__k):
            test_result = self.__dnn.test(
                self.__test_set,
                model_weight_path=self.__directory.best_weight_path(fold),
            )

            results.append(test_result)

        df = pd.DataFrame(
            results,
            index=[i for i in range(self.__k)],
            columns=self.__metrics_without_val(),
        )

        df.to_csv(self.__directory.test_result_path)

    def __metrics_without_val(self) -> List[str]:
        return list(
            filter(
                lambda name: not name.startswith("val_") and not name == "epoch",
                self.__dnn.get_metrics(),
            )
        )

    def predict(self, x: np.ndarray, fold: Optional[int] = None) -> np.ndarray:
        """
        一番良い重みで推論を行います.
        学習方法が"kcv"の場合foldを指定できます.

        Args:
            x (np.ndarray): 推論に使うデータ
            fold (Optional[int], optional): 学習方法が"kcv"のときに使う重みを選択します

        Raises:
            RuntimeError: 学習方法が"kcv"または"holdout"以外だった場合

        Returns:
            np.ndarray: 推論の結果
        """

        self.__dnn.compile()

        prediction = self.__dnn.predict(
            x, model_weight_path=self.__directory.best_weight_path(fold)
        )

        return prediction

    def plot(self) -> None:
        """
        学習履歴から結果をプロットします.
        """
        HistoryPlotter.set_style()

        for fold in range(self.__k):
            history_path = self.__directory.history_path(fold)
            history = LearningHistory.from_path(history_path)
            figure_dir = self.__directory.figure_fold_dir(fold)

            plotter = HistoryPlotter(history)
            plotter.plot_all_metrics(figure_dir)

        histories = LearningHistory.from_dir(self.__directory.histories_dir)
        average = LearningHistory.average(*histories)

        plotter = HistoryPlotter(average)
        plotter.plot_all_metrics(self.__directory.figure_average)

        test_result = LearningHistory.from_path(self.__directory.test_result_path)
        plotter = HistoryPlotter(test_result)
        plotter.box_plot(self.__directory.test_result_figure_path)
