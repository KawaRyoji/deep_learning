# DNN 実験ライブラリ

author: 川凌司

## はじめに

このライブラリは DNN における実験をなるべくわかりやすく, 簡単に行うことを目的としたプログラムです.

Tensorflow で記述されており, バージョンは 2.6 を想定して書いているので, バージョン違いによるバグがあるかもしれません.

## 使い方

```python
class SomeModel(DNN):
    @dataclass
    class Params(ModelParams): # モデルで使用するパラメータを定義
        stride: int = field(default=4)

    def definition(self, params: Params) -> Model:
        # 上で定義したモデルのパラメータからkerasのモデルを作成します
        return some_model

model = SomeModel(
    SomeModel.Params(),                     # モデルのパラメータ設定
    loss = "binary_crossentropy",           # 損失関数の設定
    optimizer = Adam(learning_rate=0.0001), # オプティマイザの設定
    metrics = ["precision"]                 # 評価値の設定
)

train_set = Dataset(train_data, train_label) # ファイルパスからデータセットを構築する方法もあります
test_set = Dataset(test_data, test_label)

param = DatasetParams(      # データセットのパラメータ
    batch_size=32,          # バッチサイズ
    epochs=100,             # エポック数
    batches_per_epoch=None  # エポック当たりのバッチ数
)

experiment = KCVExperiment(
    model,                      # 作成したモデル
    "experimental_result",      # 結果を保存するディレクトリ
    train_set,                  # 学習データセット
    test_set,                   # テストデータセット
    param,                      # データセットのパラメータ
    train_method="holdout",     # 学習方法 ホールドアウト法かk分割交差検証を選べます
    valid_split= 0.8            # データセットの分割割合 この場合学習:検証 = 8:2となります
)

experiment.train()  # 学習時に学習履歴と途中経過のチェックポイント, 評価値がもっともよい重みを保存します
experiment.test()   # テスト結果が保存されます
experiment.plot()   # 学習結果とテスト結果をプロットします
```
