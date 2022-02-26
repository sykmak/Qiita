#!/usr/bin/env python

# ライブラリ
import copy
from typing import List

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyper
import seaborn as sns


def visualize_scores(
    df: pd.DataFrame,
) -> None:
    """
    平均点推移を可視化する関数
    """

    # インデックスを取得
    indexes = df.index

    fig = plt.figure(figsize=(18, 20))

    ymin = 35
    ymax = 85

    kyoutsuu_start = 2020.5
    kyoutsuu_end = 2022.5

    ax1 = fig.add_subplot(6, 1, 1)
    ax1.plot(indexes, df["国語"], lw=2, ls="solid", marker="o", c="r", markerfacecolor="w", label="国語")
    ax1.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel("国語", fontsize=20)
    ax1.set_title("平均点", fontsize=30)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax1.legend(fontsize=15, loc="upper left")

    ax2 = fig.add_subplot(6, 1, 2)
    ax2.plot(indexes, df["世界史"], lw=2, ls="solid", marker="o", c="orange", markerfacecolor="w", label="世界史")
    ax2.plot(indexes, df["日本史"], lw=2, ls="dashed", marker="x", c="orange", markerfacecolor="w", label="日本史")
    ax2.plot(indexes, df["地理"], lw=2, ls="dotted", marker="^", c="orange", markerfacecolor="w", label="地理")
    ax2.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("地理歴史", fontsize=20)
    ax2.tick_params(axis="both", which="major", labelsize=15)
    ax2.legend(fontsize=15, loc="upper left")

    ax3 = fig.add_subplot(6, 1, 3)
    ax3.plot(indexes, df["現代社会"], lw=2, ls="solid", marker="o", c="darkgreen", markerfacecolor="w", label="現代社会")
    ax3.plot(indexes, df["倫理"], lw=2, ls="dashed", marker="x", c="darkgreen", markerfacecolor="w", label="倫理")
    ax3.plot(indexes, df["政治経済"], lw=2, ls="dotted", marker="^", c="darkgreen", markerfacecolor="w", label="政治経済")
    ax3.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax3.set_ylim(ymin, ymax)
    ax3.set_ylabel("公民", fontsize=20)
    ax3.tick_params(axis="both", which="major", labelsize=15)
    ax3.legend(fontsize=15, loc="upper left")

    ax4 = fig.add_subplot(6, 1, 4)
    ax4.plot(indexes, df["数学1A"], lw=4, ls="solid", marker="o", c="cyan", markerfacecolor="w", label="数学1A")
    ax4.plot(indexes, df["数学2B"], lw=2, ls="dashed", marker="x", c="cyan", markerfacecolor="w", label="数学2B")
    ax4.axhline(y=df.loc[2022, "数学1A"], xmin=0, xmax=1, lw=4, ls="dashed", c="cyan")
    ax4.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax4.set_ylim(ymin, ymax)
    ax4.set_ylabel("数学", fontsize=20)
    ax4.tick_params(axis="both", which="major", labelsize=15)
    ax4.legend(fontsize=15, loc="upper left")

    ax5 = fig.add_subplot(6, 1, 5)
    ax5.plot(indexes, df["物理"], lw=2, ls="solid", marker="o", c="b", markerfacecolor="w", label="物理")
    ax5.plot(indexes, df["化学"], lw=2, ls="dashed", marker="x", c="b", markerfacecolor="w", label="化学")
    ax5.plot(indexes, df["生物"], lw=2, ls="dotted", marker="^", c="b", markerfacecolor="w", label="生物")
    ax5.plot(indexes, df["地学"], lw=2, ls="dashdot", marker="s", c="b", markerfacecolor="w", label="地学")
    ax5.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax5.set_ylim(ymin, ymax)
    ax5.set_ylabel("理科", fontsize=20)
    ax5.tick_params(axis="both", which="major", labelsize=15)
    ax5.legend(fontsize=15, loc="upper left")

    ax6 = fig.add_subplot(6, 1, 6)
    ax6.plot(indexes, df["英語"], lw=2, ls="solid", marker="o", c="purple", markerfacecolor="w", label="英語")
    ax6.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax6.set_ylim(ymin, ymax)
    ax6.set_xlabel("年度", fontsize=20)
    ax6.set_ylabel("外国語", fontsize=20)
    ax6.tick_params(axis="both", which="major", labelsize=15)
    ax6.legend(fontsize=15, loc="upper left")

    # 表示
    # plt.show()

    # 保存
    fig.savefig("./fig01.png")

    return None


def visualize_correlation(
    df: pd.DataFrame,
) -> None:
    """
    標本相関係数行列を可視化する関数
    """

    # 列名を取得
    colnames = df.columns

    # 学習データ
    df = df.drop(index=2022)

    # 学習データの平均と標準偏差
    mean = df.mean()
    std = df.std()

    # 学習データを正規化
    df = (df - mean) / std

    fig = plt.figure(figsize=(18, 15))

    sns.heatmap(
        np.cov(df.T),
        vmin=-1.0,
        vmax=1.0,
        center=0,
        annot=True,  # True:格子の中に値を表示
        fmt=".2f",
        xticklabels=colnames,
        yticklabels=colnames,
        cmap="RdBu",
    )
    # 表示
    # plt.show()

    # 保存
    fig.savefig("./fig02.png")

    return None


def derive_abnormality(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    年毎の異常度を計算する関数
    """

    def pass_to_R(
        r: object,
        df_train: pd.DataFrame,
    ) -> object:
        """
        Rに処理を渡す関数
        """

        # データフレームをアサインする
        r.assign("df_train", df_train)

        # コード全文
        code = """
            library(dplyr)
            library(glasso)
        
            # 観測点数
            n_row_train <- nrow(df_train)

            # 標本分散共分散行列
            omega <- cov(df_train)

            #####################################
            # BIC最小のrhoを探す
            #####################################

            n_iter <- 91

            rho <- seq(0.10, 1.00, length=n_iter)
            bic <- rho

            # rhoの各値ごとにループ
            for(i in 1:n_iter){
            tmp_model <- glasso(omega, rho[i], nobs=n_row_train, thr=1.0e-6)
            p_off_d <- sum(tmp_model$wi != 0 & col(omega) < row(omega))
            bic[i] <- -2*(tmp_model$loglik) + p_off_d*log(n_row_train)
            }

            index_best <- which.min(bic)
            rho_and_bic <- data.frame(rho, bic)

            #####################################
            # BIC最小のグラフ構造を計算
            #####################################

            gmodel <- glasso(omega, rho[index_best], nobs=n_row_train, thr=1.0e-6)
        """

        # コード実行
        r(code)

        return r

    def cal_abnormality(
        index: int,
        pm: pd.DataFrame,
        x: pd.Series,
    ) -> float:
        """
        異常度を計算する関数
        """

        a = (
            0.5 * np.log(2 * np.pi / pm.iloc[index, index])
            + 0.5 / pm.iloc[index, index] * (np.dot(pm.iloc[index, :], x)) ** 2
        )

        return a

    def visualize_precision(
        pm: pd.DataFrame,
    ) -> None:
        """
        精度行列を可視化する関数
        """

        fig = plt.figure(figsize=(18, 15))
        sns.heatmap(
            pm,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True,  # True:格子の中に値を表示
            fmt=".2f",
            xticklabels=colnames,
            yticklabels=colnames,
            cmap="RdBu",
        )

        # plt.show()

        # 保存
        fig.savefig("./fig03.png")

        return None

    def visualize_bic(
        bic: List[float],
    ) -> None:
        """
        rho vs BICを可視化する関数
        """

        fig = plt.figure(figsize=(15, 8))

        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(bic[" rho "], bic[" bic "], lw=2, ls="solid", marker="o", c="k", markerfacecolor="w", label="BIC")
        ax1.set_xlim(0.05, 1.05)
        ax1.set_xlabel("正則化パラメータの値", fontsize=20)
        ax1.set_ylabel("BIC", fontsize=20)
        ax1.tick_params(axis="both", which="major", labelsize=15)
        # ax1.legend(fontsize=15, loc="upper left")

        # plt.show()

        # 保存
        fig.savefig("./fig04.png")

        return None

    # 列名を取得
    colnames = df.columns
    # インデックスを取得
    indexes = df.index

    # 異常度を格納するデータフレーム
    df_abnormality = df.copy(deep=True)
    # 値をリセット
    df_abnormality.loc[:, :] = np.nan

    # 年毎にループ
    for year in indexes:

        # 学習データとテストデータに分ける
        df_train = df.drop(index=year)
        df_test = df.loc[year, :]

        mean = df_train.mean()
        std = df_train.std()

        # 学習データを正規化
        df_train = (df_train - mean) / std
        # テストデータを正規化
        df_test = (df_test - mean) / std

        # Rインスタンスを生成
        r = pyper.R(use_pandas="True")

        # glassoの適用
        r = pass_to_R(r, df_train)

        # スパース化された精度行列の取得
        pm = r.get("gmodel$wi")
        # データフレーム化
        pm = pd.DataFrame(pm)

        # 2022年度のみ精度行列とBICを可視化する
        if year == 2022:
            visualize_precision(pm)
            visualize_bic(r.get("rho_and_bic"))

        # 異常度を計算
        abnormality = [cal_abnormality(index, pm, df_test) for index in range(len(colnames))]

        # 異常度を格納
        df_abnormality.loc[year, :] = abnormality

    return df_abnormality


def visualize_abnormality(
    df_abnormality: pd.DataFrame,
) -> None:
    """
    年毎の異常度を可視化する関数
    """

    # 列名を取得
    colnames = df_abnormality.columns
    # インデックスを取得
    indexes = df_abnormality.index

    #####################################
    # 異常度(2022年度のみ)
    #####################################

    fig = plt.figure(figsize=(15, 8))

    plt.bar(
        x=colnames,
        height=df_abnormality.loc[2022, :],
        width=0.5,
        alpha=0.5,
        color=[
            "r",
            "orange",
            "orange",
            "orange",
            "darkgreen",
            "darkgreen",
            "darkgreen",
            "cyan",
            "cyan",
            "b",
            "b",
            "b",
            "b",
            "purple",
        ],
        linewidth=3,
    )
    plt.ylabel("異常度", fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=15)

    # plt.show()

    # 保存
    fig.savefig("./fig05.png")

    #####################################
    # 異常度(推移)
    #####################################

    fig = plt.figure(figsize=(18, 20))

    ymin = 0
    ymax = 7

    kyoutsuu_start = 2020.5
    kyoutsuu_end = 2022.5

    ax1 = fig.add_subplot(6, 1, 1)
    ax1.plot(indexes, df_abnormality["国語"], lw=2, ls="solid", marker="o", c="r", markerfacecolor="w", label="国語")
    ax1.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel("国語", fontsize=20)
    ax1.set_title("異常度", fontsize=30)
    ax1.tick_params(axis="both", which="major", labelsize=15)
    ax1.legend(fontsize=15, loc="upper left")

    ax2 = fig.add_subplot(6, 1, 2)
    ax2.plot(
        indexes, df_abnormality["世界史"], lw=2, ls="solid", marker="o", c="orange", markerfacecolor="w", label="世界史"
    )
    ax2.plot(
        indexes, df_abnormality["日本史"], lw=2, ls="dashed", marker="x", c="orange", markerfacecolor="w", label="日本史"
    )
    ax2.plot(indexes, df_abnormality["地理"], lw=2, ls="dotted", marker="^", c="orange", markerfacecolor="w", label="地理")
    ax2.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("地理歴史", fontsize=20)
    ax2.tick_params(axis="both", which="major", labelsize=15)
    ax2.legend(fontsize=15, loc="upper left")

    ax3 = fig.add_subplot(6, 1, 3)
    ax3.plot(
        indexes, df_abnormality["現代社会"], lw=2, ls="solid", marker="o", c="darkgreen", markerfacecolor="w", label="現代社会"
    )
    ax3.plot(
        indexes, df_abnormality["倫理"], lw=2, ls="dashed", marker="x", c="darkgreen", markerfacecolor="w", label="倫理"
    )
    ax3.plot(
        indexes,
        df_abnormality["政治経済"],
        lw=2,
        ls="dotted",
        marker="^",
        c="darkgreen",
        markerfacecolor="w",
        label="政治経済",
    )
    ax3.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax3.set_ylim(ymin, ymax)
    ax3.set_ylabel("公民", fontsize=20)
    ax3.tick_params(axis="both", which="major", labelsize=15)
    ax3.legend(fontsize=15, loc="upper left")

    ax4 = fig.add_subplot(6, 1, 4)
    ax4.plot(
        indexes, df_abnormality["数学1A"], lw=4, ls="solid", marker="o", c="cyan", markerfacecolor="w", label="数学1A"
    )
    ax4.plot(
        indexes, df_abnormality["数学2B"], lw=2, ls="dashed", marker="x", c="cyan", markerfacecolor="w", label="数学2B"
    )
    ax4.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax4.set_ylim(ymin, ymax)
    ax4.set_ylabel("数学", fontsize=20)
    ax4.tick_params(axis="both", which="major", labelsize=15)
    ax4.legend(fontsize=15, loc="upper left")

    ax5 = fig.add_subplot(6, 1, 5)
    ax5.plot(indexes, df_abnormality["物理"], lw=2, ls="solid", marker="o", c="b", markerfacecolor="w", label="物理")
    ax5.plot(indexes, df_abnormality["化学"], lw=2, ls="dashed", marker="x", c="b", markerfacecolor="w", label="化学")
    ax5.plot(indexes, df_abnormality["生物"], lw=2, ls="dotted", marker="^", c="b", markerfacecolor="w", label="生物")
    ax5.plot(indexes, df_abnormality["地学"], lw=2, ls="dashdot", marker="s", c="b", markerfacecolor="w", label="地学")
    ax5.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax5.set_ylim(ymin, ymax)
    ax5.set_ylabel("理科", fontsize=20)
    ax5.tick_params(axis="both", which="major", labelsize=15)
    ax5.legend(fontsize=15, loc="upper left")

    ax6 = fig.add_subplot(6, 1, 6)
    ax6.plot(indexes, df_abnormality["英語"], lw=2, ls="solid", marker="o", c="purple", markerfacecolor="w", label="英語")
    ax6.fill_between([kyoutsuu_start, kyoutsuu_end], ymin, ymax, color="lightgray")
    ax6.set_ylim(ymin, ymax)
    ax6.set_xlabel("年度", fontsize=20)
    ax6.set_ylabel("外国語", fontsize=20)
    ax6.tick_params(axis="both", which="major", labelsize=15)
    ax6.legend(fontsize=15, loc="upper left")

    # plt.show()

    # 保存
    fig.savefig("./fig06.png")


if __name__ == "__main__":

    # センター試験平均点
    df = pd.read_csv("./CenterExamAverageScores.csv", encoding="utf-8", index_col=0)
    # 使わない列を落とす
    df = df.drop(columns=["リスニング"])

    # 平均点の推移の可視化
    visualize_scores(df)
    # 標本相関係数の可視化
    visualize_correlation(df)
    # 年毎の異常度の計算
    df_abnormality = derive_abnormality(df)
    # 年毎の異常度の推移の可視化
    visualize_abnormality(df_abnormality)
