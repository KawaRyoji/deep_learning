import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import numpy as np
import tensorflow as tf
from deep_learning.plot import HistoryPlotter
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint

from deep_learning.dataset import Dataset, DatasetParams
from deep_learning.dnn import DNN, CheckPoint, CheckPointCallBack, LearningHistory


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

    def __post_init__(self):
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


class DNNExperiment:
    """
    DNNを用いた実験を行うクラスです.
    """

    def __init__(
        self,
        dnn: DNN,
        root_dir: str,
        train_set: Dataset,
        test_set: Dataset,
        dataset_params: DatasetParams,
        train_method: str = "holdout",
        k: Optional[int] = None,
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
            model_params (dict, optional): モデルのパラメータ
            train_method (str, optional): 学習方法("holdout" または "kcv")
            k (Optional[int], optional): k分割交差検証のパラメータ
            valid_split (Optional[float], optional): 検証用データの割合
            gpu (Optional[int], optional): 使用するGPU番号

        Raises:
            RuntimeError: train_method が "kcv" かつ k を指定していない場合
            RuntimeError: train_method が "holdout" または "kcv" でない場合
        """
        self.__set_hardware(gpu_i=gpu)

        if train_method == "holdout":
            self.__directory: Union[HoldoutDirectory, KCVDirectory] = HoldoutDirectory(
                root_dir
            )
        elif train_method == "kcv":
            if k is None:
                raise RuntimeError("k must not be None.")
            self.__directory = KCVDirectory(root_dir)
            self.__k = k
        else:
            raise RuntimeError("Train method assumes 'holdout' or 'kcv'.")

        self.__dnn = dnn
        self.__train_method = train_method
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
        """
        datasetの保存先パスからインスタンスを生成します.

        Args:
            dnn (DNN): DNNモデル
            root_dir (str): 結果保存先のディレクトリパス
            train_set_path (str): 学習用データセットのパス
            test_set_path (str): テスト用データセットのパス
            dataset_params (DatasetParams):  データセットのパラメータ
            model_params (dict, optional):  モデルのパラメータ
            train_method (str, optional):  学習方法("holdout" または "kcv")
            k (Optional[int], optional):  k分割交差検証のパラメータ
            valid_split (Optional[float], optional):  検証用データの割合
            gpu (Optional[int], optional):  使用するGPU番号

        Returns:
            DNNExperiment: 生成したインスタンス
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
        clear_session()  # メモリリーク対策
        if self.__train_method == "holdout":
            self.__holdout_train(
                use_checkpoint=use_checkpoint,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                additional_callbacks=additional_callbacks,
            )
        elif self.__train_method == "kcv":
            self.__kcv_train(
                use_checkpoint=use_checkpoint,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                additional_callbacks=additional_callbacks,
            )

    def __holdout_train(
        self,
        use_checkpoint: bool = True,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """
        ホールドアウト法でモデルを学習させます.

        Args:
            from_checkpoint (bool, optional): チェックポイントから学習を再開させるかどうか
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

        checkpoint = self.__load_ho_checkpoint(
            use_checkpoint, self.__dataset_params.epochs
        )

        self.__dnn.train(
            train_sequence=train_sequence,
            valid_sequence=valid_sequence,
            epochs=self.__dataset_params.epochs,
            checkpoint=checkpoint,
            callbacks=callbacks,
        )

        self.__used_checkpoint(directory.history_path)

    def __kcv_train(
        self,
        use_checkpoint: bool = True,
        monitor_metric="val_loss",
        monitor_mode="auto",
        additional_callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """
        k分割交差検証でモデルを学習させます.

        Args:
            use_checkpoint (bool, optional): 学習を再開させるためのチェックポイントのパス
            monitor_metric (str, optional): モデルの評価値　この評価値で性能の良い重みを決定します
            monitor_mode (str, optional): モデルの評価値の監視モード
            additional_callbacks (Optional[List[Callback]], optional): 追加するコールバック関数
        """
        directory: KCVDirectory = self.__directory

        sequence = self.__train_set.to_kcv_data_sequence(
            self.__dataset_params.batch_size,
            self.__k,
            batches_per_epoch=self.__dataset_params.batches_per_epoch,
        )

        checkpoint = self.__load_kcv_checkpoint(
            use_checkpoint, self.__dataset_params.epochs
        )

        if checkpoint is None:
            init_fold = 0
        else:
            init_fold = checkpoint.fold

        for fold, train_sequence, valid_sequence in sequence.generate():
            if init_fold > fold:  # チェックポイントに記述されたfoldまで飛ばす
                continue

            clear_session()  # メモリリーク対策
            self.__dnn.compile()

            callbacks = [
                CSVLogger(directory.history_path(fold)),
                CheckPointCallBack(
                    directory.checkpoint_path,
                    directory.latest_weight_path,
                    fold=fold,
                ),
                ModelCheckpoint(
                    directory.best_weight_path(fold),
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
                checkpoint=checkpoint,
                callbacks=callbacks,
            )

            # チェックポイントから始めた場合に元の学習履歴が消去されてしまう問題を解決する
            self.__used_checkpoint(self.__directory.history_path(fold))
            checkpoint = None  # 一度チェックポイントを使ったら使わない

    def __load_ho_checkpoint(
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
        tmp_history.save_to(os.path.join(self.__directory.histories_dir, "tmp.csv"))

        return checkpoint

    def __used_checkpoint(self, new_history_path: str) -> None:
        if os.path.exists(os.path.join(self.__directory.histories_dir, "tmp.csv")):
            old_history = LearningHistory.from_path(
                os.path.join(self.__directory.histories_dir, "tmp.csv")
            )
            new_history = LearningHistory.from_path(new_history_path)
            history = LearningHistory.concat([old_history, new_history])
            history.save_to(new_history_path)
            os.remove(os.path.join(self.__directory.histories_dir, "tmp.csv"))

    def __load_kcv_checkpoint(
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

        if self.__train_method == "holdout":
            self.__holdout_test()
        elif self.__train_method == "kcv":
            self.__kcv_test()

    def __holdout_test(self) -> None:
        """
        ホールドアウト法でのテストを行います.
        """
        directory: HoldoutDirectory = self.__directory
        test_sequence = self.__test_set.to_data_sequence(1, shuffle=False)

        test_result = self.__dnn.test(
            test_sequence,
            model_weight_path=directory.best_weight_path,
        )

        history = LearningHistory.from_list([test_result], self.__metrics_without_val())
        history.save_to(directory.test_result_path)

    def __kcv_test(self) -> None:
        """
        k分割交差検証でのテストを行います.
        """
        directory: KCVDirectory = self.__directory
        test_sequence = self.__test_set.to_data_sequence(1, shuffle=False)

        results = []
        for fold in range(self.__k):
            test_result = self.__dnn.test(
                test_sequence,
                model_weight_path=directory.best_weight_path(fold),
            )

            results.append(test_result)

        df = pd.DataFrame(
            results,
            index=[i for i in range(self.__k)],
            columns=self.__metrics_without_val(),
        )

        df.to_csv(directory.test_result_path)

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

        if self.__train_method == "kcv":
            if fold is None:
                fold = 0
            prediction = self.__dnn.predict(
                x, model_weight_path=self.__directory.best_weight_path(fold)
            )
        elif self.__train_method == "holdout":
            prediction = self.__dnn.predict(
                x, model_weight_path=self.__directory.best_weight_path
            )
        else:
            raise RuntimeError("train method must be 'kcv' or 'holdout'")

        return prediction

    def plot(self):
        """
        学習履歴から結果をプロットします.
        """
        HistoryPlotter.set_style()

        if self.__train_method == "holdout":
            self.__plot_holdout()
        elif self.__train_method == "kcv":
            self.__plot_kcv()

    def __plot_holdout(self):
        """
        ホールドアウト法でのプロットを行います.
        """
        directory: HoldoutDirectory = self.__directory
        history = LearningHistory.from_path(directory.history_path)

        plotter = HistoryPlotter(history)
        plotter.plot_all_metrics(directory.figures_dir)

    def __plot_kcv(self):
        """
        k分割交差検証でのプロットを行います.
        """
        directory: KCVDirectory = self.__directory
        for fold in range(self.__k):
            history_path = directory.history_path(fold)
            history = LearningHistory.from_path(history_path)
            figure_dir = directory.figure_fold_dir(fold)

            plotter = HistoryPlotter(history)
            plotter.plot_all_metrics(figure_dir)

        histories = LearningHistory.from_dir(directory.histories_dir)
        average = LearningHistory.average(*histories)

        plotter = HistoryPlotter(average)
        plotter.plot_all_metrics(directory.figure_average)

        test_result = LearningHistory.from_path(directory.test_result_path)
        plotter = HistoryPlotter(test_result)
        plotter.box_plot(directory.test_result_figure_path)
