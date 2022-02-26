#!/usr/bin/env python

# ライブラリ
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_testquantity(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    データフレームを受け取り、標本分散と統計検定量の列を追加して返す関数
    """

    # 標本分散
    df["variance"] = (
        df["支持率"] / 100 * (1 - df["支持率"] / 100)
        + df["不支持率"] / 100 * (1 - df["不支持率"] / 100)
        + 2 * df["支持率"] / 100 * df["不支持率"] / 100
    ) / df["n"]

    # 統計検定量
    df["z"] = (df["支持率"] / 100 - df["不支持率"] / 100) / np.sqrt(df["variance"])

    return df


def figure1(
    df: pd.DataFrame,
) -> None:
    """
    図1（調査年月 vs 支持率、不支持率）を出力する関数
    """

    # figureインスタンスの生成
    fig = plt.figure(figsize=(15, 5))

    # プロットする列を取得
    x = df["year-month"]
    y_approve = df["支持率"]
    y_disapprove = df["不支持率"]

    # 支持率
    plt.plot(
        x,
        y_approve,
        lw=2,
        linestyle="-",
        color="r",
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="r",
        label="支持率",
    )
    # 不支持率
    plt.plot(
        x,
        y_disapprove,
        lw=2,
        linestyle="-",
        color="b",
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="b",
        label="不支持率",
    )

    # ラベル
    plt.xlabel("調査年月", fontsize=20)
    plt.ylabel("支持率, 不支持率 (%)", fontsize=20)

    # ticksのサイズ
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # グリッド
    plt.grid(True)

    # 凡例
    plt.legend(fontsize=20)

    # タイトル
    # plt.title('菅内閣支持率・不支持率', fontsize=20)

    # 表示
    # plt.show()

    # 保存
    fig.savefig("./fig01.jpeg")

    return None


def figure2(
    df: pd.DataFrame,
) -> None:
    """
    図2（調査年月 vs 支持率-不支持率）を出力する関数
    """

    # figureインスタンスの生成
    fig = plt.figure(figsize=(15, 5))

    # プロットする列を取得
    x = df["year-month"]
    y_approve = df["支持率"]
    y_disapprove = df["不支持率"]

    # 支持率-不支持率
    plt.plot(
        x,
        y_approve - y_disapprove,
        lw=2,
        linestyle="-",
        color="k",
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="k",
        label="支持率-不支持率",
    )

    # ラベル
    plt.xlabel("調査年月", fontsize=20)
    plt.ylabel("支持率 - 不支持率 (%)", fontsize=20)

    # y=0の線
    plt.hlines(y=0, xmin=0, xmax=12, lw=2, linestyle="--", color="k")

    # ticksのサイズ
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # グリッド
    plt.grid(True)

    # 凡例
    # plt.legend(fontsize=20)

    # 表示
    # plt.show()

    # 保存
    fig.savefig("./fig02.jpeg")

    return None


def figure3(
    df: pd.DataFrame,
) -> None:
    """
    図3（調査年月 vs 支持率-不支持率; 誤差付き）を出力する関数
    """

    # 標本分散と統計検定量を取得
    df = get_testquantity(df)

    # エラーバーの情報を追加
    y_err = 1.645 * np.sqrt(df["variance"]) * 100

    # figureインスタンスの生成
    fig = plt.figure(figsize=(15, 5))

    # プロットする列を取得
    x = df["year-month"]
    y_approve = df["支持率"]
    y_disapprove = df["不支持率"]

    # 支持率-不支持率
    plt.errorbar(
        x,
        y_approve - y_disapprove,
        yerr=y_err,
        lw=2,
        linestyle="-",
        color="k",
        fmt="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="k",
        capsize=5,
        ecolor="k",
        label="支持率-不支持率",
    )

    # 拡大用
    # plt.xlim(4.5,8.5)
    # plt.ylim(-12,12)

    # ラベル
    plt.xlabel("調査年月", fontsize=20)
    plt.ylabel("支持率 - 不支持率 (%)", fontsize=20)

    # y=0の線
    plt.hlines(y=0, xmin=0, xmax=12, lw=2, linestyle="--", color="k")

    # ticksのサイズ
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # グリッド
    plt.grid(True)

    # 凡例
    # plt.legend(fontsize=20)

    # 表示
    # plt.show()

    # 保存
    fig.savefig("./fig03.jpeg")

    return None


if __name__ == "__main__":

    # データ読み込み
    df = pd.read_csv("./SugaCabinetApprovalRate.csv")
    # 年月のカラムを追加
    df["year-month"] = df["年"].astype(str) + "/" + df["月"].astype(str)

    figure1(df)  # 図1
    figure2(df)  # 図2
    figure3(df)  # 図3, 4
