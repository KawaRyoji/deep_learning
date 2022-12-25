import gc
import os
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
import tracemalloc
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Adam, Optimizer

from deep_learning.dataset import DataSequence
from deep_learning.util import JSONObject


@dataclass
class ModelParams(JSONObject):
    """
    DNNモデルのパラメータを定義するクラスです.
    """

    def to_dict(self) -> dict:
        return asdict(self)


class DNN(metaclass=ABCMeta):
    """
    DNNモデルを定義する抽象クラスです.
    使用するにはモデルを定義するdefinition関数をオーバーロードしてください.
    """

    def __init__(
        self,
        param: ModelParams,
        loss: Union[str, Loss],
        optimizer: Union[str, Optimizer] = Adam(),
        metrics: List[Union[str, Metric]] = None,
    ) -> None:
        """
        Args:
            loss (Union[str, Loss]): 損失関数
            optimizer (Union[str, Optimizer], optional): オプティマイザ
            metrics (List[Union[str, Metric]], optional): 評価値
        """
        self.loss = loss
        self.param = param
        self.optimizer = optimizer
        self.metrics = metrics
        self.__model: tf.keras.Model = None

    @abstractmethod
    def definition(self, param: ModelParams) -> Model:
        """
        モデルを定義する関数です.

        Raises:
            NotImplementedError: オーバーライドされていないとき

        Returns:
            Model: 定義したモデル
        """
        raise NotImplementedError()

    def get_metrics(self) -> List[str]:
        """
        モデルに設定されているmetricを返します.

        Returns:
            List[str]: モデルに設定されているmetricのリスト
        """
        self.__ensure_model_compiled()

        return self.__model.metrics_names

    def compile(self) -> None:
        """
        モデルをコンパイルします.
        """
        if self.__model is not None:
            del self.__model

        self.__model = self.definition(self.param)
        self.__model.compile(
            loss=self.loss, metrics=self.metrics, optimizer=self.optimizer
        )

    def train(
        self,
        train_dataset: tf.data.Dataset,
        epochs: int,
        valid_dataset: Optional[
            Union[tf.data.Dataset, Tuple[np.ndarray, np.ndarray]]
        ] = None,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        checkpoint: Optional["CheckPoint"] = None,
        callbacks: List[Callback] = None,
    ) -> None:
        """
        モデルを学習させます.

        Args:
            train_dataset (tf.data.Dataset): 学習データ
            epochs (int): エポック数
            valid_dataset (tf.data.Dataset, optional): 検証データ
            check_point (CheckPoint, optional): 学習を再開するチェックポイント
            callbacks (List[Callback], optional): コールバック関数
        """
        self.__ensure_model_compiled()

        init_epoch = 0
        if checkpoint is not None:
            init_epoch = checkpoint.epoch + 1  # checkpoint.epochの次から開始させるため+1
            self.load(checkpoint.weight_path)

        self.__model.fit(
            x=train_dataset,
            validation_data=valid_dataset,
            epochs=epochs,
            initial_epoch=init_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )


    def test(
        self, test_sequence: DataSequence, model_weight_path: str = None
    ) -> np.ndarray:
        """
        モデルでテストを行います.

        Args:
            test_sequence (DataSequence): テストデータのシークエンス
            model_weight_path (str, optional): モデルの重みのパス

        Returns:
            np.ndarray: _description_
        """
        self.__ensure_model_compiled()

        if model_weight_path is not None:
            self.load(model_weight_path)

        return self.__model.evaluate(x=test_sequence, verbose=2)

    def predict(self, x: np.ndarray, model_weight_path: str = None) -> np.ndarray:
        """
        モデルで推論を行います.

        Args:
            x (np.ndarray): 推論を行うデータ
            model_weight_path (str, optional): モデルの重みのパス

        Returns:
            np.ndarray: 推論結果
        """
        self.__ensure_model_compiled()

        if model_weight_path is not None:
            self.load(model_weight_path)

        return self.__model.predict(x)

    def load(self, model_weight_path: str) -> None:
        """
        モデルの重みのパスから重みを読み込みます.

        Args:
            model_weight_path (str): モデルの重みのパス
        """
        self.__ensure_model_compiled()

        self.__model.load_weights(model_weight_path)

    def __ensure_model_compiled(self) -> None:
        if self.__model is None:
            raise RuntimeError("Model should be compiled.")


@dataclass
class CheckPoint(JSONObject):
    """
    モデルの途中経過を保存するクラスです.
    """

    weight_path: str
    epoch: int
    timestamp: datetime = field(default=datetime.now())
    fold: Optional[int] = field(default=None)

    @classmethod
    def load(cls, path: str) -> Union["CheckPoint", None]:
        """
        jsonファイルのパスからインスタンスを生成します.

        Args:
            path (str): jsonファイルのパス

        Returns:
            CheckPoint | None: 生成したインスタンス、パスが存在しなければNone
        """
        if not os.path.exists(path):
            return None
        return super().load(path, object_hook=cls.__object_hook)

    def dump(self, path: str) -> None:
        """
        オブジェクトをjsonファイルに保存します.

        Args:
            path (str): 保存するjsonファイルのパス
        """
        super().dump(path, default=self.__json_default)

    @staticmethod
    def __json_default(obj):
        return obj.isoformat() if hasattr(obj, "isoformat") else obj

    @staticmethod
    def __object_hook(obj):
        dic = dict()
        for o in obj:
            try:
                dic[str(o)] = datetime.strptime(obj[o], "%Y-%m-%dT%H:%M:%S.%f")
            except Exception:
                dic[str(o)] = obj[o]
        return dic


class CheckPointCallBack(ModelCheckpoint):
    """
    チェックポイントを学習中に生成するコールバック関数です.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_weight_path: str,
        fold: Optional[int] = None,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        options=None,
        **kwargs
    ) -> None:
        """
        Args:
            checkpoint_path (str): チェックポイントのパス
            model_weight_path (str): 保存しておくモデルの重みのパス
            fold (Optional[int], optional): fold
            その他ModelCheckpointのパラメータ
        """
        super().__init__(
            model_weight_path,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            options,
            **kwargs
        )

        self.checkpoint_path = checkpoint_path
        self.fold = fold

    def on_epoch_end(self, epoch, logs=None) -> None:
        super().on_epoch_end(epoch, logs=logs)

        if os.path.exists(self.filepath):
            cp = CheckPoint(self.filepath, epoch, fold=self.fold)
            cp.dump(self.checkpoint_path)

class GCCallback(Callback):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None) -> None:
        gc.collect()
    
class LearningHistory:
    """
    tensorflowのCSVLoggerで保存された学習履歴を扱うクラスです。
    """

    def __init__(self, df_history: pd.DataFrame, metrics: List[str]) -> None:
        """
        Args:
            df_history (pd.DataFrame): 学習履歴のcsvを読み込んだデータフレーム
            metrics (list): 扱う評価値
        """
        self.df_history = df_history
        self.metrics = metrics

    @classmethod
    def from_path(cls, history_path: str) -> "LearningHistory":
        """
        学種履歴のcsvのパスを指定してインスタンスを生成します

        Args:
            history_path (str): 学習履歴のcsvのパス

        Returns:
            History: 生成したインスタンス
        """
        df_history = pd.read_csv(history_path, index_col=0)
        metrics = df_history.columns.to_list()

        return cls(df_history, metrics)

    @classmethod
    def from_dir(cls, history_dir: str) -> List["LearningHistory"]:
        """
        学習履歴が保存されているディレクトリから、リスト形式でインスタンスを生成します。

        Returns:
            List[History]: ディレクトリに含まれる学習履歴から生成したインスタンスのリスト
        """
        from deep_learning.util import dir2paths

        history_paths = dir2paths(history_dir)
        histories = list(
            map(lambda history_path: cls.from_path(history_path), history_paths)
        )

        return histories

    @classmethod
    def from_list(
        cls, history_list: List[Any], metrics: List[str]
    ) -> "LearningHistory":
        history = pd.DataFrame(history_list, columns=metrics)

        return cls(history, metrics)

    @classmethod
    def concat(cls, histories: List["LearningHistory"]) -> "LearningHistory":
        df_histories = list(map(lambda history: history.df_history, histories))
        df_history = pd.concat(df_histories)

        return cls(df_history, df_history.columns.to_list())

    @classmethod
    def average(cls, *learning_histories: "LearningHistory") -> "LearningHistory":
        """
        複数の学習履歴から、各評価値ごとに平均を計算し、学習履歴のインスタンスを返します。

        Returns:
            History: 平均された学習履歴のインスタンス
        """
        df_histories = list(map(lambda x: x.df_history, learning_histories))
        df_avg = sum(df_histories) / len(df_histories)
        metrics = df_avg.columns.to_list()

        return cls(df_avg, metrics)

    def filter_by_metrics(self, metrics: List[str]) -> "LearningHistory":
        """
        指定した評価値でフィルタリングします。

        Args:
            metrics (List[str]): フィルタリングしたい評価値

        Returns:
            History: 指定した評価値でフィルタリングされた学習履歴のインスタンス
        """
        filtered = LearningHistory(self.df_history.filter(items=metrics), metrics)

        return filtered

    def melt(self) -> pd.DataFrame:
        """
        横持ちのデータを縦持ちに変換したデータフレームを返します。
        箱ひげ図のプロットに必要となります。

        Returns:
            pd.DataFrame: 変換したデータフレーム
        """
        return pd.melt(self.df_history, ignore_index=True)

    def of_metric(self, metric: str) -> pd.DataFrame:
        """
        指定した評価値のデータフレームを返します。

        Args:
            metric (str): 指定する評価値

        Returns:
            pd.DataFrame: 指定した評価値に対応するデータフレーム
        """
        return self.df_history[metric]

    def save_to(self, file_path: str) -> None:
        """
        学習履歴を指定したパスに保存します。

        Args:
            file_path (str): 保存先のパス(csvファイル)
        """
        pd.DataFrame.to_csv(self.df_history, file_path)

    def __str__(self) -> str:
        return self.df_history.__str__()
