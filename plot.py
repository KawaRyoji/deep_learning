import os
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from prototype.dnn import LearningHistory


def graph_settings(
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[str] = None,
    ylim: Optional[str] = None,
    legend: Optional[str] = None,
    fig_path: Optional[str] = None,
    legend_loc: str = "best",
    tick: bool = True,
    grid: bool = True,
    close: bool = False,
    show: bool = False,
) -> None:
    if title is not None:
        plt.title(title, fontsize=16)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=16)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=16)

    if tick:
        plt.tick_params(labelsize=14)

    if grid:
        plt.grid()

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    if legend is not None:
        plt.legend(legend, loc=legend_loc, fontsize=14)

    if fig_path is not None:
        Path.mkdir(Path(fig_path).parent, parents=True, exist_ok=True)
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0.2)

    if show:
        plt.show()

    if close:
        plt.close()


class HistoryPlotter:
    def __init__(self, history: LearningHistory, style="white", palette="Set1") -> None:
        self.history = history
        sns.set()
        sns.set_style(style=style)
        sns.set_palette(palette=palette)

    def plot(self, metric: str, path: str) -> None:
        train_history = self.history.of_metric(metric)
        valid_history = self.history.of_metric("val_" + metric)

        ylim = (
            min(train_history.min(), valid_history.min()),
            max(train_history.max(), valid_history.max()),
        )

        plt.figure()
        train_history.plot()
        valid_history.plot()

        label = metric.replace("_", " ")

        graph_settings(
            xlabel="epoch",
            ylabel=label,
            xlim=(0, train_history.size - 1),
            ylim=ylim,
            fig_path=path,
            legend=["Training", "Validation"],
            close=True,
        )

    def plot_all_metrics(self, dir_path: str) -> None:
        metrics = list(
            filter(lambda c: (not c.startswith("val_")), self.history.metrics)
        )

        for metric in metrics:
            self.plot(metric, os.path.join(dir_path, metric + ".png"))

    def box_plot(
        self, path: str, stripplot: bool = False, metrics: Optional[List[str]] = None
    ) -> None:
        if metrics is None:
            metrics = list(filter(lambda c: not c == "loss", self.history.metrics))

        filtered = self.history.filter_by_metrics(metrics)
        melted = filtered.melt()

        fig, ax = plt.subplots()
        sns.boxplot(x="variable", y="value", data=melted, whis=[0, 100], ax=ax)

        if stripplot:
            sns.stripplot(
                x="variable", y="value", data=melted, jitter=True, color="black", ax=ax
            )

        graph_settings(xlabel="metrics", ylabel="score", fig_path=path, close=True)

    @staticmethod
    def comparison_plot(
        metric: str,
        path: str,
        *histories: LearningHistory,
        legend: Optional[List[str]] = None
    ) -> None:
        if legend is not None:
            if len(legend) != len(histories):
                raise RuntimeError("historiesとlegendの次元を一致させてください")

        xlim = (
            0,
            max(list(map(lambda history: len(history.of_metric(metric)), histories)))
            - 1,
        )  # 学習履歴の最大のエポック数を計算
        ylim = (
            min(list(map(lambda history: history.of_metric(metric).min(), histories))),
            max(list(map(lambda history: history.of_metric(metric).max(), histories))),
        )  # 学習履歴の評価値の最小、最大を計算

        label = label[4:] if metric.startswith("val_") else metric
        label = metric.replace("_", " ")

        plt.figure()
        for history in histories:
            history.of_metric(metric).plot()

        graph_settings(
            xlabel="epoch",
            ylabel=label,
            xlim=xlim,
            ylim=ylim,
            fig_path=path,
            legend=legend,
            close=True,
        )

    @staticmethod
    def comparison_plot_all_metrics(
        dir_path: str, *histories: LearningHistory, legend: Optional[List[str]] = None
    ) -> None:
        metrics = histories[0].metrics

        for metric in metrics:
            HistoryPlotter.comparison_plot(
                metric,
                os.path.join(dir_path, metric + ".png"),
                *histories,
                legend=legend
            )

    @staticmethod
    def comparison_box_plot(
        path: str,
        *histories: LearningHistory,
        stripplot=False,
        metrics: Optional[List[str]] = None,
        legend: Optional[List[str]] = None
    ) -> None:
        if legend is None:
            legend = ["history{}".format(i + 1) for i in range(len(histories))]
        else:
            if len(legend) != len(histories):
                raise "history_pathsとlegendの次元を一致させてください"

        if metrics is None:
            metrics = histories[0].metrics
            metrics = list(filter(lambda c: not c == "loss"))

        filtered_histories = list(
            map(lambda history: history.filter_by_metrics(metrics), histories)
        )

        melted_histories = list(map(lambda history: history.melt(), filtered_histories))

        for melted_history, l in zip(melted_histories, legend):
            melted_history["group"] = l

        df = pd.concat(melted_histories, axis=0)

        fig, ax = plt.subplots()
        sns.boxplot(x="variable", y="value", data=df, hue="group", whis=[0, 100], ax=ax)

        h, l = ax.get_legend_handles_labels()
        if stripplot:
            sns.stripplot(
                x="variable",
                y="value",
                data=df,
                hue="group",
                dodge=True,
                jitter=True,
                color="black",
                ax=ax,
            )
            h, l = ax.get_legend_handles_labels()
            h = h[: len(h) // 2]
            l = l[: len(l) // 2]

        ax.legend(h, l)
        graph_settings(
            xlabel="metrics",
            ylabel="score",
            fig_path=path,
            close=True,
        )


class ActivationPlotter:
    def __init__(self, activation: np.ndarray) -> None:
        self.activation = activation

    def plot(self, path: str, color_map: str = "magma") -> None:
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.ioff()

        plt.set_cmap(color_map)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, np.flipud(self.activation))
        plt.close()

    def plot_with_label(
        self, label: np.ndarray, path: str, color_map: str = "magma"
    ) -> None:
        fig, ax = plt.subplots(
            dpi=100,
            figsize=(self.activation.shape[1] / 100, self.activation.shape[0] / 100),
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        ax.pcolor(self.activation, cmap=color_map)
        label_color = mpl.colors.ListedColormap([(0, 0, 0, 0), (0, 1, 1, 0.5)])
        ax.pcolor(label, cmap=label_color)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        plt.savefig(path)
        plt.close()
