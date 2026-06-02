"""
品目×活動種別 ROI 分析スクリプト
====================================
activity_data.csv（活動ログ）と delivery_data.csv（施設別月次納入）を結合し、
パネルOLS回帰（施設固定効果）で活動種別ごとの限界納入寄与額を推定する。

ROI = 活動1回あたり限界納入額(万円) / 活動1回あたりMRコスト(万円)

出力:
  output/roi_by_product_activity.csv   品目×活動種別ROIテーブル
  output/roi_heatmap.html              ROIヒートマップ（Plotly）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_ROOT = Path(__file__).parent.parent
BASE_DIR = _ROOT / "data" / "input"
OUTPUT_DIR = _ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# パラメータ
# ============================================================

# MR1人あたり年間コスト（万円）。人件費+経費+間接費 概算。
MR_ANNUAL_COST_MAN_YEN = 1_500

# 年間稼働日数 × 1日あたり時間
MR_WORKING_HOURS_PER_YEAR = 240 * 8   # 1,920h

# 活動種別ごとのMR所要時間（時間/件）
# ※ 説明会・Web講演会はイベント準備も含む担当者の実工数を医師接触数で割った値
ACTIVITY_COST_HOURS: dict[str, float] = {
    "面談":      0.50,
    "面談_アポ": 0.70,
    "説明会":    0.25,   # グループイベント: 全体2.5h ÷ 平均10人
    "Web講演会": 0.20,   # グループイベント: 全体4h  ÷ 平均20人
    "メール":    0.08,
    "Web面談":   0.45,
}

# MR時給（万円/時間）
MR_HOURLY_COST = MR_ANNUAL_COST_MAN_YEN / MR_WORKING_HOURS_PER_YEAR

# 活動種別ごとのコスト（万円/件）
ACTIVITY_COST_MAN_YEN = {
    k: round(v * MR_HOURLY_COST, 4)
    for k, v in ACTIVITY_COST_HOURS.items()
}

# ROI推定対象の活動種別
ACTIVITY_TYPES = list(ACTIVITY_COST_HOURS.keys())


# ============================================================
# データ読み込み
# ============================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    act = pd.read_csv(BASE_DIR / "activity_data.csv", encoding="utf-8-sig")
    deli = pd.read_csv(BASE_DIR / "delivery_data.csv", encoding="utf-8-sig")
    return act, deli


# ============================================================
# 前処理
# ============================================================

def prepare_panel(
    act: pd.DataFrame,
    deli: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    品目ごとに (delivery_ym × facility_id) のパネルデータを作成する。
    説明変数: 各活動種別の月次件数
    被説明変数: 月次納入金額（百万円）
    """
    # activity_date → activity_ym
    act = act.copy()
    act["activity_ym"] = act["activity_date"].str[:7]

    # 活動件数を (ym, facility_id, product_id, activity_type) で集計
    act_agg = (
        act.groupby(["activity_ym", "facility_id", "product_id", "activity_type"])
        .size()
        .reset_index(name="activity_count")
    )

    # ワイド形式に変換: 列 = activity_type
    act_wide = act_agg.pivot_table(
        index=["activity_ym", "facility_id", "product_id"],
        columns="activity_type",
        values="activity_count",
        fill_value=0,
    ).reset_index()
    act_wide.columns.name = None

    # 存在しない活動種別列を0で補完
    for at in ACTIVITY_TYPES:
        if at not in act_wide.columns:
            act_wide[at] = 0

    # 品目ごとにパネル作成
    panels: dict[str, pd.DataFrame] = {}
    for pid in act_wide["product_id"].unique():
        a = act_wide[act_wide["product_id"] == pid].copy()
        d = deli[deli["product_id"] == pid][
            ["delivery_ym", "facility_id", "amount_mn"]
        ].copy()

        panel = d.merge(a, left_on=["delivery_ym", "facility_id"],
                        right_on=["activity_ym", "facility_id"], how="left")
        panel["activity_ym"] = panel["delivery_ym"]

        for at in ACTIVITY_TYPES:
            if at not in panel.columns:
                panel[at] = 0
            else:
                panel[at] = panel[at].fillna(0)

        panels[pid] = panel

    return panels


# ============================================================
# OLS with facility fixed effects (within estimator)
# ============================================================

def ols_within(
    panel: pd.DataFrame,
    dep_var: str = "amount_mn",
    indep_vars: list[str] = ACTIVITY_TYPES,
) -> dict[str, float]:
    """
    施設固定効果 OLS（within推定）を numpy で実装する。
    各変数から施設平均を引いたデータで OLS を実行。
    戻り値: {変数名: 係数} ← 活動1件あたり追加納入額（百万円）
    """
    df = panel[["facility_id", dep_var] + indep_vars].dropna().copy()
    if len(df) < 10:
        return {v: 0.0 for v in indep_vars}

    # within変換（施設平均を引く）
    df["_fid"] = df["facility_id"]
    grp = df.groupby("_fid")
    df[dep_var + "_dm"] = df[dep_var] - grp[dep_var].transform("mean")
    for v in indep_vars:
        df[v + "_dm"] = df[v] - grp[v].transform("mean")

    Y = df[dep_var + "_dm"].values
    X = df[[v + "_dm" for v in indep_vars]].values

    # Ridge正則化（微小値でほぼOLS）
    alpha = 1e-6
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ Y
    try:
        coefs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coefs = np.zeros(len(indep_vars))

    return {v: float(c) for v, c in zip(indep_vars, coefs)}


# ============================================================
# ROI 計算
# ============================================================

def calc_roi(coef_mn: float, activity_type: str) -> float:
    """
    ROI = 限界納入額（万円/件） / 活動コスト（万円/件）
    coef_mn: OLS係数（百万円/件）→ 100万円 = 100万円
    """
    marginal_man_yen = coef_mn * 100  # 百万円 → 万円
    cost_man_yen = ACTIVITY_COST_MAN_YEN[activity_type]
    if cost_man_yen <= 0:
        return 0.0
    return round(marginal_man_yen / cost_man_yen, 2)


# ============================================================
# メイン
# ============================================================

def main() -> None:
    print("データ読み込み中...")
    act, deli = load_data()
    print(f"  活動データ: {len(act):,} 行")
    print(f"  納入データ: {len(deli):,} 行")

    print("パネルデータ作成中...")
    panels = prepare_panel(act, deli)

    rows: list[dict] = []
    for pid, panel in panels.items():
        print(f"  OLS推定: {pid} (n={len(panel):,}行)")
        coefs = ols_within(panel)
        for at in ACTIVITY_TYPES:
            c = coefs.get(at, 0.0)
            roi = calc_roi(c, at)
            rows.append({
                "product_id":          pid,
                "activity_type":       at,
                "coef_mn_per_act":     round(c, 6),     # 百万円/件
                "marginal_man_yen":    round(c * 100, 2),  # 万円/件
                "cost_man_yen":        ACTIVITY_COST_MAN_YEN[at],
                "roi":                 roi,
            })

    roi_df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "roi_by_product_activity.csv"
    roi_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[出力] {out_csv}")

    # ---- ヒートマップ ----
    pivot = roi_df.pivot(index="product_id", columns="activity_type", values="roi")
    # 活動種別の列順を固定
    col_order = [c for c in ACTIVITY_TYPES if c in pivot.columns]
    pivot = pivot[col_order]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="品目: %{y}<br>活動種別: %{x}<br>ROI: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="品目×活動種別 ROI（施設固定効果OLS推定）",
        xaxis_title="活動種別",
        yaxis_title="品目",
        width=900,
        height=700,
    )
    out_html = OUTPUT_DIR / "roi_heatmap.html"
    fig.write_html(str(out_html))
    print(f"[出力] {out_html}")

    # ---- サマリ表示 ----
    print("\n===== 品目×活動種別 ROI =====")
    summary = roi_df.pivot(
        index="product_id", columns="activity_type", values="roi"
    )[col_order]
    print(summary.to_string(float_format="{:.2f}".format))

    print("\n===== 活動種別ごとの平均ROI（全品目） =====")
    avg = roi_df.groupby("activity_type")["roi"].mean().reindex(ACTIVITY_TYPES)
    for at, r in avg.items():
        cost = ACTIVITY_COST_MAN_YEN[at]
        print(f"  {at:12s}: ROI={r:6.2f}  (コスト {cost:.4f} 万円/件)")


if __name__ == "__main__":
    main()
