"""
FY2029 FTE算出 HTML レポート生成モジュール
==========================================
算出結果をインタラクティブなHTMLレポートとして出力する。
Plotlyを使用（Databricks環境で標準的に利用可能）。
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# カラーパレット（品目別）
# ============================================================

PRODUCT_COLORS: Dict[str, str] = {
    # PDT群 ── 青系
    "GLI": "#1f77b4",   # 中青
    "GLO": "#5bc8af",   # ティール（GLIの一時代替品）
    "CUV": "#17becf",   # シアン
    "HYQ": "#6baed6",   # 薄青
    # NS群 ── 赤/橙系（大品目: 目立つ色）
    "INT": "#d62728",   # 赤
    "TRI": "#ff7f0e",   # 橙
    "ENT": "#e6550d",   # 深橙
    # OVE ── マゼンタ/ピンク
    "OVE": "#e377c2",   # マゼンタ
    # CV群 ── 緑系
    "LIV": "#2ca02c",   # 緑
    "REV": "#74c476",   # 明るい緑
    "ALC": "#006837",   # 深緑
    # RS群 ── 紫系
    "VYV": "#9467bd",   # 紫
    "VPR": "#7b2d8b",   # 深紫
    # NEW群 ── 黄/オリーブ系（CS新規発売品）
    "Zaso": "#bcbd22",  # オリーブ黄
    "WSA":  "#ffdd44",  # 明黄
    # PS群 ── 茶/水色系
    "LVM": "#8c564b",   # 茶
    "TKZ": "#4393c3",   # スチールブルー
    "RPL": "#92c5de",   # 薄水色
    "VON": "#f4a582",   # サーモン
}

GROUP_COLORS = {
    "PDT": "#1f77b4",
    "NS":  "#d62728",
    "OVE": "#e377c2",
    "NEW": "#bcbd22",
    "CV":  "#2ca02c",
    "RS":  "#9467bd",
    "PS":  "#8c564b",
}


# ============================================================
# ユーティリティ
# ============================================================

def _fig_to_html(fig: go.Figure, div_id: str = "") -> str:
    """Plotly FigureをHTMLの<div>文字列に変換（埋め込み用）"""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": True, "responsive": True},
    )


def _df_to_html(
    df: pd.DataFrame,
    title: str = "",
    highlight_cols: Optional[List[str]] = None,
    bar_cols: Optional[List[str]] = None,
) -> str:
    """DataFrameをスタイル付きHTMLテーブルに変換"""
    styler = df.style

    if bar_cols:
        styler = styler.bar(subset=bar_cols, color="#a8d8ea", vmin=0)

    if highlight_cols:
        styler = styler.background_gradient(
            subset=highlight_cols, cmap="YlOrRd", low=0, high=1
        )

    styler = (
        styler
        .format(precision=2)
        .set_table_attributes('class="styled-table"')
    )

    html = styler.to_html()
    if title:
        html = f'<h3 class="table-title">{title}</h3>' + html
    return html


# ============================================================
# グラフ生成
# ============================================================

def fig_fte_by_product_bar(summary_df: pd.DataFrame, target_fy: str = "") -> go.Figure:
    """
    品目別 平均必要FTE（MRのみ）の棒グラフ。FTE=MR headcountのみ。
    summary_df: summarize_fy() の出力
    """
    # FTE降順でソート
    df = summary_df.copy().sort_values("avg_required_fte", ascending=False)
    fy_label = f" - {target_fy}" if target_fy else ""

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="必要FTE（MR）",
        x=df["product_id"],
        y=df["avg_required_fte"],
        marker_color=[PRODUCT_COLORS.get(p, "#aaa") for p in df["product_id"]],
        text=df["avg_required_fte"].round(1),
        textposition="outside",
    ))

    fig.update_layout(
        title=f"品目別 平均必要FTE（MR headcount）{fy_label}",
        xaxis_title="品目",
        yaxis_title="FTE（人）",
        height=400,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def fig_monthly_fte_trend(fte_df: pd.DataFrame) -> go.Figure:
    """
    品目×月別 FTE 推移: 左=絶対値ライン、右=100%積み上げエリア
    """
    fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("品目別 月次FTE推移（絶対値）", "品目別 月次FTE構成比（100%積み上げ）"),
        horizontal_spacing=0.10,
    )

    products = sorted(fte_df["product_id"].unique())

    # 月×品目のピボット（100%積み上げ用）
    pivot = (
        fte_df.groupby(["month", "product_id"])[fte_col]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=products, fill_value=0)
    )
    total_by_month = pivot.sum(axis=1).replace(0, np.nan)
    pct_pivot = pivot.div(total_by_month, axis=0).fillna(0) * 100

    for pid in products:
        sub = fte_df[fte_df["product_id"] == pid].sort_values("month")
        color = PRODUCT_COLORS.get(pid, "#aaa")
        has_fte = bool(sub[fte_col].sum() > 0)

        # 左: 絶対値ライン
        fig.add_trace(
            go.Scatter(
                name=pid,
                x=sub["month"],
                y=sub[fte_col],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=has_fte,
            ),
            row=1, col=1,
        )

        # 右: 100%積み上げ
        if pid in pct_pivot.columns and pct_pivot[pid].sum() > 0:
            fig.add_trace(
                go.Scatter(
                    name=pid,
                    x=pct_pivot.index.tolist(),
                    y=pct_pivot[pid].round(1).tolist(),
                    mode="lines",
                    stackgroup="pct",
                    fillcolor=color,
                    line=dict(color=color, width=0.5),
                    showlegend=False,
                ),
                row=1, col=2,
            )

    fig.update_layout(
        height=460,
        template="plotly_white",
        title="月次FTE推移",
        legend=dict(orientation="v", y=0.5, x=1.02, font=dict(size=11)),
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="FTE（人）", row=1, col=1)
    fig.update_yaxes(title_text="構成比（%）", range=[0, 100], row=1, col=2)
    return fig


def fig_digital_activity(digital_act_df: pd.DataFrame) -> go.Figure:
    """
    デジタル活動（webinar/e_contents）の品目別視聴数チャート。
    視聴ログのみのデータ: 全行が視聴イベント。

    左: 品目別 視聴数（webinar/e_contents 積み上げ）
    右: 活動種別ごとの品目シェア（100%積み上げ）
    """
    if digital_act_df.empty:
        return go.Figure()

    df = digital_act_df.copy()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("品目別 デジタル視聴数（全期間）", "品目別 活動種別構成比"),
        horizontal_spacing=0.12,
    )

    by_type = (
        df.groupby(["product_id", "activity_type"])
        .size()
        .reset_index(name="view_count")
    )
    products_sorted = (
        by_type.groupby("product_id")["view_count"].sum()
        .sort_values(ascending=False).index.tolist()
    )

    # 左: 品目別視聴数（webinar/e_contents 積み上げ）
    for act_type, color in [("webinar", "#2563eb"), ("e_contents", "#16a34a")]:
        sub = (
            by_type[by_type["activity_type"] == act_type]
            .set_index("product_id")
            .reindex(products_sorted, fill_value=0)
            .reset_index()
        )
        fig.add_trace(go.Bar(
            name=act_type,
            x=sub["product_id"],
            y=sub["view_count"],
            marker_color=color,
            text=sub["view_count"],
            textposition="inside",
        ), row=1, col=1)

    # 右: 100%積み上げ（活動種別構成比）
    total_by_prod = by_type.groupby("product_id")["view_count"].sum()
    for act_type, color in [("webinar", "#2563eb"), ("e_contents", "#16a34a")]:
        sub = (
            by_type[by_type["activity_type"] == act_type]
            .set_index("product_id")
            .reindex(products_sorted, fill_value=0)
            .reset_index()
        )
        pct = (sub["view_count"] / total_by_prod.reindex(products_sorted, fill_value=1).values * 100).round(1)
        fig.add_trace(go.Bar(
            name=act_type,
            x=sub["product_id"],
            y=pct,
            marker_color=color,
            text=pct.astype(str) + "%",
            textposition="inside",
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(
        barmode="stack",
        height=420,
        template="plotly_white",
        title="デジタル活動（webinar / e_contents）視聴ログ",
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_yaxes(title_text="視聴数", row=1, col=1)
    fig.update_yaxes(title_text="構成比（%）", range=[0, 110], row=1, col=2)
    fig.update_xaxes(tickangle=45)
    return fig


def fig_digital_trend(digital_act_df: pd.DataFrame) -> go.Figure:
    """品目別 月次デジタル視聴数推移（webinar/e_contents別）"""
    if digital_act_df.empty:
        return go.Figure()

    df = digital_act_df.copy()
    if "activity_date" in df.columns and "activity_ym" not in df.columns:
        df["activity_ym"] = df["activity_date"].str[:7]
    if "activity_ym" not in df.columns:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("webinar 月次視聴数", "e_contents 月次視聴数"),
        horizontal_spacing=0.10,
    )

    for col_idx, act_type in enumerate(["webinar", "e_contents"], start=1):
        sub_df = df[df["activity_type"] == act_type]
        monthly = (
            sub_df.groupby(["activity_ym", "product_id"])
            .size()
            .reset_index(name="view_count")
        )
        for pid in sorted(monthly["product_id"].unique()):
            s = monthly[monthly["product_id"] == pid].sort_values("activity_ym")
            color = PRODUCT_COLORS.get(pid, "#aaa")
            fig.add_trace(go.Scatter(
                name=pid,
                x=s["activity_ym"],
                y=s["view_count"],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=(col_idx == 1),
            ), row=1, col=col_idx)

    fig.update_layout(
        height=380,
        template="plotly_white",
        title="品目別 月次デジタル視聴数推移",
        legend=dict(orientation="v", y=0.5, x=1.02, font=dict(size=10)),
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="視聴数")
    return fig


def fig_mr_digital_ratio(summary_df: pd.DataFrame, target_fy: str = "") -> go.Figure:
    """品目別 MR/デジタル比率 水平バー"""
    df = summary_df.copy().sort_values("avg_mr_ratio")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="MR比率",
        y=df["product_id"],
        x=df["avg_mr_ratio"] * 100,
        orientation="h",
        marker_color="#1f77b4",
        text=(df["avg_mr_ratio"] * 100).round(0).astype(int).astype(str) + "%",
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="デジタル比率",
        y=df["product_id"],
        x=df["avg_digital_ratio"] * 100,
        orientation="h",
        marker_color="#ff7f0e",
        text=(df["avg_digital_ratio"] * 100).round(0).astype(int).astype(str) + "%",
        textposition="inside",
    ))

    fy_label = f" - {target_fy}" if target_fy else ""
    fig.update_layout(
        barmode="stack",
        title=f"品目別 MR / デジタル活動比率{fy_label}<br><sup>MMMチャネル限界効果（1回あたり増分）× ライフサイクルで算出</sup>",
        xaxis_title="比率（%）",
        xaxis=dict(range=[0, 100]),
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


def fig_per_launch_allocation(per_launch_allocations: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    新発売品ごとの発売時点FTE配分チャート（OVE/Zaso/WSA各々の発売時にどの品目から何FTE削減するか）。
    サブプロット: 発売品ごとに水平バーチャート。
    """
    products = [p for p in ["OVE", "Zaso", "WSA"] if p in per_launch_allocations]
    if not products:
        return go.Figure()

    launch_labels = {"OVE": "OVE（FY2026-07発売）", "Zaso": "Zaso（FY2027-04発売）", "WSA": "WSA（FY2029-04発売）"}

    fig = make_subplots(
        rows=1, cols=len(products),
        subplot_titles=[launch_labels[p] for p in products],
        horizontal_spacing=0.10,
    )

    for col_idx, pid in enumerate(products, start=1):
        df = per_launch_allocations[pid].copy().sort_values("fte_reduction", ascending=True)
        fig.add_trace(go.Bar(
            name=pid,
            y=df["product_id"],
            x=df["fte_reduction"],
            orientation="h",
            marker_color=[PRODUCT_COLORS.get(p, "#aaa") for p in df["product_id"]],
            text=df["fte_reduction"].round(1),
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        title="新発売品の発売時点FTE配分（品目別削減量）",
        height=350,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="削減FTE（人）")
    return fig


def fig_ove_allocation(allocation_df: pd.DataFrame) -> go.Figure:
    """OVE新発売FTE配分先（ドナー品目）の水平バー"""
    df = allocation_df.copy().sort_values("fte_reduction", ascending=True)

    fig = go.Figure(go.Bar(
        y=df["product_id"],
        x=df["fte_reduction"],
        orientation="h",
        marker_color=[PRODUCT_COLORS.get(p, "#aaa") for p in df["product_id"]],
        text=df["fte_reduction"].round(1),
        textposition="outside",
        customdata=df[["marginal_roi", "current_fte", "new_fte"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "削減FTE: %{x:.1f}<br>"
            "現在FTE: %{customdata[1]:.1f} → 調整後: %{customdata[2]:.1f}<br>"
            "限界ROI: %{customdata[0]:.6f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="新発売品（OVE/Zaso/WSA）に伴うFTE配分（どの品目から削るか）",
        xaxis_title="削減FTE（人）",
        height=320,
        template="plotly_white",
    )
    return fig


def fig_raw_fte_by_fy(raw_summary_fy: pd.DataFrame) -> go.Figure:
    """
    正規化前の本来必要FTE（活動積み上げ・制約なし）を年度別に表示。
    CS/PSそれぞれの本来必要FTEと現行ヘッドカウントを比較し、
    実際の充足率・不足数を可視化する。
    """
    from fy2029_fte_calculator import CURRENT_MR_COUNT

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CS領域 本来必要FTE vs 現行MR数", "PS領域 本来必要FTE vs 現行MR数"),
        horizontal_spacing=0.12,
    )

    area_col = {"CS": 1, "PS": 2}
    area_colors = {"CS": "#1f77b4", "PS": "#ff7f0e"}

    for area in ["CS", "PS"]:
        col = area_col[area]
        sub = raw_summary_fy[raw_summary_fy["area"] == area].sort_values("fiscal_year")
        if sub.empty:
            continue
        current = CURRENT_MR_COUNT.get(area, 0)

        # 本来必要FTE バー
        fig.add_trace(go.Bar(
            name=f"{area} 本来必要FTE",
            x=sub["fiscal_year"],
            y=sub["avg_total_fte"].round(0),
            marker_color=area_colors[area],
            text=sub["avg_total_fte"].round(0).astype(int),
            textposition="outside",
            showlegend=True,
        ), row=1, col=col)

        # 現行MR数（点線）
        fig.add_trace(go.Scatter(
            name=f"{area} 現行MR数 ({current}名)",
            x=sub["fiscal_year"],
            y=[current] * len(sub),
            mode="lines+text",
            line=dict(color="#d62728", width=3, dash="dash"),
            text=[str(current)] * len(sub),
            textposition="top center",
            showlegend=True,
        ), row=1, col=col)

    fig.update_layout(
        height=440,
        template="plotly_white",
        title="本来必要FTE（活動から積み上げ・正規化前）vs 現行MR数",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        margin=dict(b=120),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def fig_fte_vs_headcount(total_fte_df: pd.DataFrame) -> go.Figure:
    """領域別 合計FTE vs 現行MR数の月別比較"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("CS領域", "PS領域"),
    )

    for col_idx, area in enumerate(["CS", "PS"], start=1):
        sub = total_fte_df[total_fte_df["area"] == area].sort_values("month")
        if sub.empty:
            continue

        # 必要FTE
        fig.add_trace(go.Scatter(
            name=f"{area} 必要FTE",
            x=sub["month"],
            y=sub["total_fte"],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
        ), row=1, col=col_idx)

        # 現行MR数（水平線）
        fig.add_trace(go.Scatter(
            name=f"{area} 現行MR数",
            x=sub["month"],
            y=sub["current_mr"],
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dash"),
        ), row=1, col=col_idx)

        # GAP領域を塗る
        fig.add_trace(go.Scatter(
            name=f"{area} GAP",
            x=pd.concat([sub["month"], sub["month"].iloc[::-1]]),
            y=pd.concat([sub["total_fte"], sub["current_mr"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="rgba(255,0,0,0)"),
            showlegend=False,
        ), row=1, col=col_idx)

    # derive year range from data for title
    fy_range_label = ""
    if "fiscal_year" in total_fte_df.columns:
        fy_vals = sorted(total_fte_df["fiscal_year"].unique())
        if fy_vals:
            fy_range_label = f" ({fy_vals[0]}〜{fy_vals[-1]})"
    fig.update_layout(
        title=f"必要FTE vs 現行MR数比較{fy_range_label}",
        height=400,
        template="plotly_white",
    )
    fig.update_xaxes(tickangle=45)
    return fig


def fig_fy_trend(summary_df: pd.DataFrame) -> go.Figure:
    """
    品目×年度の平均FTE推移（グループ別積み上げ棒グラフ）
    summary_df: summarize_fy() の出力（fiscal_year カラム必須）
    """
    from fy2029_fte_calculator import PRODUCT_GROUPS

    if "fiscal_year" not in summary_df.columns:
        return go.Figure()

    # 品目→グループのマッピング
    pid_to_group = {
        pid: grp
        for grp, pids in PRODUCT_GROUPS.items()
        for pid in pids
    }
    summary_df = summary_df.copy()
    summary_df["group"] = summary_df["product_id"].map(pid_to_group).fillna("OTHER")

    fy_list = sorted(summary_df["fiscal_year"].unique())
    products = sorted(summary_df["product_id"].unique())

    fig = go.Figure()
    for pid in products:
        sub = summary_df[summary_df["product_id"] == pid].set_index("fiscal_year")
        y_vals = [sub.loc[fy, "avg_required_fte"] if fy in sub.index else 0.0 for fy in fy_list]
        fig.add_trace(go.Bar(
            name=pid,
            x=fy_list,
            y=y_vals,
            marker_color=PRODUCT_COLORS.get(pid, "#aaa"),
            text=[f"{v:.1f}" if v > 0 else "" for v in y_vals],
            textposition="inside",
        ))

    fy_range = f"{fy_list[0]}〜{fy_list[-1]}" if fy_list else ""
    fig.update_layout(
        barmode="stack",
        title=f"品目別 年度FTE推移（{fy_range}）",
        xaxis_title="年度",
        yaxis_title="平均必要FTE（人）",
        height=560,
        template="plotly_white",
        legend=dict(
            orientation="h",
            y=-0.28,
            x=0.5,
            xanchor="center",
            traceorder="normal",
        ),
        margin=dict(b=160),
    )
    return fig


def fig_fy_trend_area(total_fte_fy_df: pd.DataFrame) -> go.Figure:
    """
    領域×年度の合計FTE推移（折れ線 + 現行MR数比較）
    total_fte_fy_df: total_fte_by_area_fy() の出力
    """
    fy_vals_area = sorted(total_fte_fy_df["fiscal_year"].unique()) if "fiscal_year" in total_fte_fy_df.columns else []
    area_fy_range = f"（{fy_vals_area[0]}〜{fy_vals_area[-1]}）" if fy_vals_area else ""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CS領域", "PS領域"))

    for col_idx, area in enumerate(["CS", "PS"], start=1):
        sub = total_fte_fy_df[total_fte_fy_df["area"] == area].sort_values("fiscal_year")
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=f"{area} 必要FTE",
            x=sub["fiscal_year"],
            y=sub["avg_total_fte"],
            marker_color="#1f77b4" if area == "CS" else "#ff7f0e",
            text=sub["avg_total_fte"].round(1),
            textposition="outside",
        ), row=1, col=col_idx)
        fig.add_trace(go.Scatter(
            name=f"{area} 現行MR数",
            x=sub["fiscal_year"],
            y=sub["current_mr"],
            mode="lines+markers",
            line=dict(color="#d62728", dash="dash", width=2),
        ), row=1, col=col_idx)

    fig.update_layout(
        title=f"領域別 年度FTE vs 現行MR数{area_fy_range}",
        height=380,
        template="plotly_white",
    )
    return fig


def fig_sim_vs_optimal(
    summary_df: pd.DataFrame,
    optimal_fte_df: pd.DataFrame,
) -> go.Figure:
    """
    シミュレーションFTE vs ROI最適FTEの品目別比較チャート。

    左: 品目別 FY別 シミュレーション vs 最適FTE（FY2029フォーカス）
    右: エリア別 年度合計 シミュレーション vs 最適FTE推移
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "品目別 シミュレーションFTE vs ROI最適FTE（FY2029）",
            "エリア別 年度合計FTE 比較",
        ),
        horizontal_spacing=0.10,
    )

    # ---- 左: FY2029 品目別比較（グループ棒グラフ）----
    TARGET_FY = "FY2029"

    if not optimal_fte_df.empty and "fiscal_year" in summary_df.columns:
        sim_fy = summary_df[summary_df["fiscal_year"] == TARGET_FY].copy()
        opt_fy = optimal_fte_df[optimal_fte_df["fiscal_year"] == TARGET_FY].copy()

        # FY2029に存在する品目（FTE>0）に絞り、ROI最適降順でソート
        opt_vals_map = opt_fy.set_index("product_id")["optimal_fte"].to_dict()
        products = sorted(
            [p for p in sim_fy["product_id"].unique() if sim_fy.loc[sim_fy["product_id"] == p, "avg_required_fte"].sum() > 0],
            key=lambda p: opt_vals_map.get(p, 0.0),
            reverse=True,
        )

        sim_vals = [
            sim_fy.loc[sim_fy["product_id"] == p, "avg_required_fte"].sum()
            for p in products
        ]
        opt_vals = [
            opt_vals_map.get(p, 0.0)
            for p in products
        ]

        fig.add_trace(go.Bar(
            name=f"シミュレーション ({TARGET_FY})",
            x=products,
            y=sim_vals,
            marker_color=[PRODUCT_COLORS.get(p, "#aaa") for p in products],
            text=[f"{v:.1f}" for v in sim_vals],
            textposition="outside",
            offsetgroup="sim",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=f"ROI最適 ({TARGET_FY})",
            x=products,
            y=opt_vals,
            marker_color=[PRODUCT_COLORS.get(p, "#aaa") for p in products],
            opacity=0.5,
            text=[f"{v:.1f}" for v in opt_vals],
            textposition="outside",
            marker_pattern_shape="/",
            offsetgroup="opt",
        ), row=1, col=1)

    # ---- 右: エリア別年度合計 シミュレーション vs 最適FTE ----
    if not optimal_fte_df.empty and "fiscal_year" in summary_df.columns:
        # シミュレーション: エリア×FY合計
        sim_area = (
            summary_df.groupby(["area", "fiscal_year"])["avg_required_fte"]
            .sum()
            .reset_index()
            .rename(columns={"avg_required_fte": "total_fte"})
        )
        # 最適: エリア×FY合計
        opt_area = (
            optimal_fte_df.groupby(["area", "fiscal_year"])["optimal_fte"]
            .sum()
            .reset_index()
            .rename(columns={"optimal_fte": "total_fte"})
        )

        area_styles = {
            "CS": {"sim_color": "#1f77b4", "opt_color": "#aec7e8"},
            "PS": {"sim_color": "#ff7f0e", "opt_color": "#ffbb78"},
        }
        for area, styles in area_styles.items():
            sim_sub = sim_area[sim_area["area"] == area].sort_values("fiscal_year")
            opt_sub = opt_area[opt_area["area"] == area].sort_values("fiscal_year")

            if not sim_sub.empty:
                fig.add_trace(go.Scatter(
                    name=f"{area} シミュレーション",
                    x=sim_sub["fiscal_year"],
                    y=sim_sub["total_fte"],
                    mode="lines+markers+text",
                    line=dict(color=styles["sim_color"], width=3),
                    marker=dict(size=10),
                    text=sim_sub["total_fte"].round(0).astype(int),
                    textposition="top center",
                ), row=1, col=2)

            if not opt_sub.empty:
                fig.add_trace(go.Scatter(
                    name=f"{area} ROI最適",
                    x=opt_sub["fiscal_year"],
                    y=opt_sub["total_fte"],
                    mode="lines+markers+text",
                    line=dict(color=styles["opt_color"], width=3, dash="dash"),
                    marker=dict(size=10, symbol="diamond"),
                    text=opt_sub["total_fte"].round(0).astype(int),
                    textposition="bottom center",
                ), row=1, col=2)

    fig.update_layout(
        barmode="group",
        height=480,
        template="plotly_white",
        title="シミュレーションFTE vs ROI最大化最適FTE（売上最大化目的）",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(b=120),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def fig_fc_sc_breakdown(fte_df: pd.DataFrame) -> go.Figure:
    """FC/SC 医師数の品目別積み上げ（FY2029平均）"""
    avg = (
        fte_df.groupby("product_id")[["fc_doctors", "sc_doctors"]]
        .mean()
        .reset_index()
        .sort_values("fc_doctors", ascending=False)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="FC医師数（主訪問）",
        x=avg["product_id"],
        y=avg["fc_doctors"],
        marker_color="#1f77b4",
        text=avg["fc_doctors"].round(0).astype(int),
        textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="SC医師数（セカンドコール）",
        x=avg["product_id"],
        y=avg["sc_doctors"],
        marker_color="#aec7e8",
        text=avg["sc_doctors"].round(0).astype(int),
        textposition="inside",
    ))

    fig.update_layout(
        barmode="stack",
        title="品目別 FC / SC 医師数（全期間月平均）",
        xaxis_title="品目",
        yaxis_title="医師数",
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", y=1.05),
    )
    return fig


# ============================================================
# HTMLレポート生成
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FTE算出レポート ({target_fy})</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {{
      --primary: #2c3e50;
      --accent:  #3498db;
      --bg:      #f5f7fa;
      --card:    #ffffff;
      --border:  #dee2e6;
      --gap-pos: #e74c3c;
      --gap-neg: #27ae60;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", "Hiragino Sans", "Meiryo", sans-serif;
      background: var(--bg);
      color: var(--primary);
      font-size: 14px;
    }}
    header {{
      background: var(--primary);
      color: white;
      padding: 24px 32px;
    }}
    header h1 {{ font-size: 22px; font-weight: 700; }}
    header .meta {{ font-size: 12px; margin-top: 6px; opacity: 0.75; }}

    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}

    /* KPI バナー */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px 20px;
      text-align: center;
    }}
    .kpi-card .label {{ font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
    .kpi-card .value {{ font-size: 28px; font-weight: 700; color: var(--accent); margin: 6px 0; }}
    .kpi-card .sub   {{ font-size: 11px; color: #999; }}

    /* セクション */
    section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    section h2 {{
      font-size: 16px;
      font-weight: 700;
      border-left: 4px solid var(--accent);
      padding-left: 10px;
      margin-bottom: 18px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    .chart-full {{ grid-column: 1 / -1; }}

    /* テーブル */
    .styled-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 10px;
    }}
    .styled-table th {{
      background: var(--primary);
      color: white;
      padding: 8px 12px;
      text-align: left;
      font-weight: 600;
      white-space: nowrap;
    }}
    .styled-table td {{
      padding: 7px 12px;
      border-bottom: 1px solid var(--border);
    }}
    .styled-table tr:hover td {{ background: #f0f4ff; }}
    .table-title {{
      font-size: 14px;
      font-weight: 600;
      margin: 16px 0 6px;
      color: var(--primary);
    }}

    /* GAP バッジ */
    .gap-pos {{ color: var(--gap-pos); font-weight: 700; }}
    .gap-neg {{ color: var(--gap-neg); font-weight: 700; }}

    /* フッター */
    footer {{
      text-align: center;
      font-size: 11px;
      color: #aaa;
      padding: 24px;
    }}

    @media (max-width: 900px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
      .kpi-row {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>

<header>
  <h1>FTE算出レポート（{target_fy} フォーカス）</h1>
  <div class="meta">
    生成日時: {generated_at} ／
    対象期間: {fy_range_display}／
    頻度モード: {frequency_mode}
  </div>
</header>

<div class="container">

  <!-- ロジック概要 -->
  <section style="background:linear-gradient(135deg,#1a3560 0%,#2563eb 100%);color:#fff;border:none;">
    <h2 style="border-left-color:#93c5fd;color:#fff;">最適FTE推定ロジック ― このレポートの読み方</h2>
    <p style="font-size:13px;line-height:1.8;margin-bottom:18px;opacity:0.92;">
      本システムは「CS {current_cs}名 / PS {current_ps}名」という固定ヘッドカウントの中で、
      <strong>各品目への月別MR投入量（FTE）が最大売上を生むよう配分</strong>することを目的とします。<br>
      活動実績・品目属性・MMMパラメータを入力とし、以下の7ステップで最適FTEを算出します。
    </p>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-bottom:10px;">
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 1</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">ターゲット医師数の算出</div>
        <div style="font-size:12px;opacity:0.82;">活動実績から施設カバレッジを推定し、品目ごとの訪問対象医師数を決定</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 2</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">FC / SC 分割</div>
        <div style="font-size:12px;opacity:0.82;">医師被り率で主訪問（FC）と同行訪問（SC）に分割。SCコスト = FC × 0.1</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 3</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">訪問頻度の推定</div>
        <div style="font-size:12px;opacity:0.82;">実績ベース頻度にライフサイクル補正（発売初期×1.3 / LOE前後×0.7→0.3）を適用</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 4</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">MR / デジタル比率</div>
        <div style="font-size:12px;opacity:0.82;">MMMの減衰パラメータ（Adstock半減期・Hill応答曲線）の限界応答比からMR担当割合を推定</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 5</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">FTE 算出（コア計算）</div>
        <div style="font-size:12px;opacity:0.82;">FTE = ターゲット医師数 × 訪問頻度 × MR比率 ÷（稼働日 × コール/日）</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 6</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">新発売品への FTE 移動</div>
        <div style="font-size:12px;opacity:0.82;">OVE / Zaso / WSA の必要FTEを、限界ROIの低い既存品目から比例配分で削出</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 7</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">正規化・ROI最適配分</div>
        <div style="font-size:12px;opacity:0.82;">ヘッドカウント制約でスケーリング後、等限界収益配分（Hill関数の傾きが均等になる点）で最終配分を決定</div>
      </div>
    </div>
    <p style="font-size:11px;opacity:0.6;text-align:right;margin-top:4px;">
      詳細ロジックは fte_logic_document.html を参照
    </p>
  </section>

  <!-- KPI バナー -->
  <div class="kpi-row">
    {kpi_cards}
  </div>

  <!-- Section 0: FY トレンド（FY2026〜FY2029）-->
  <section>
    <h2>① 年度別FTE推移（FY2026〜FY2029）</h2>
    <div class="chart-grid">
      <div>{chart_fy_trend}</div>
      <div>{chart_fy_area}</div>
    </div>
    {table_fy_summary}
  </section>

  <!-- Section 0b: ROI最適FTE比較 -->
  <section>
    <h2>② ROI最大化 最適FTE vs シミュレーションFTE</h2>
    <p style="font-size:13px;color:#666;margin-bottom:12px;">
      「最適FTE」= MMMのHill関数に基づく等限界収益配分。売上最大化を目的として各品目への活動投資効率が均等になるよう配分した理論値。
      シミュレーションFTEとの差がリソース再配分の余地を示す。
    </p>
    <div class="chart-full">{chart_sim_vs_optimal}</div>
  </section>

  <!-- Section 1: FTE サマリー -->
  <section>
    <h2>③ 品目別 年度FTEサマリー（FY2029）</h2>
    <div class="chart-grid">
      <div>{chart_fte_bar}</div>
      <div>{chart_mr_digital}</div>
    </div>
    {table_summary}
  </section>

  <!-- Section 2: 月次推移 -->
  <section>
    <h2>④ 月次FTE推移</h2>
    <div class="chart-full">{chart_monthly}</div>
  </section>

  <!-- Section 3: FC/SC 分析 -->
  <section>
    <h2>⑤ ファーストコール / セカンドコール分析</h2>
    <div class="chart-grid">
      <div>{chart_fc_sc}</div>
      <div>{table_fc_sc}</div>
    </div>
  </section>

  <!-- Section 4: 新品目 FTE 配分 -->
  <section>
    <h2>⑥ 新発売品（OVE/Zaso/WSA）に伴うFTE配分</h2>
    <p style="font-size:13px;color:#666;margin-bottom:12px;">
      各新製品の発売時点で、どの既存品目から何FTEを移動するかを示す。限界ROI（収益効率）の低い品目から優先的に削減。
    </p>
    <div class="chart-full">{chart_per_launch}</div>
    <div class="chart-grid">
      <div>{chart_ove}</div>
      <div>{table_ove}</div>
    </div>
  </section>

  <!-- Section 4b: 本来必要FTE（正規化前）-->
  <section>
    <h2>⑦ 本来必要FTE（活動積み上げ・制約なし）</h2>
    <p style="font-size:13px;color:#666;margin-bottom:12px;">
      各品目の活動量（訪問頻度 × ターゲット医師数 × コール時間）から積み上げた理論上の必要FTE。
      CS=380・PS=45 のヘッドカウント制約を適用する前の数値。現行人員との差分が本来の過不足を示す。
    </p>
    <div class="chart-full">{chart_raw_fte}</div>
  </section>

  <!-- Section 5: 現行MR数との比較 -->
  <section>
    <h2>⑧ 必要FTE vs 現行MR数（{current_cs}名CS / {current_ps}名PS）</h2>
    <div class="chart-full">{chart_headcount}</div>
    {table_headcount}
  </section>

  <!-- Section 5b: デジタル活動 -->
  <section>
    <h2>⑨ デジタル活動実績（webinar / e_contents）</h2>
    <p style="font-size:13px;color:#666;margin-bottom:12px;">
      MR活動（activity_data）とは独立したデジタルチャネルの実績。
      webinar（Web講演会）・e_contents（電子コンテンツ）の品目別視聴状況。MR比率計算の参考指標。
    </p>
    <div class="chart-full">{chart_digital_activity}</div>
    <div class="chart-full">{chart_digital_trend}</div>
    {table_digital_summary}
  </section>

  <!-- Section 6: 詳細テーブル（全品目×全月）-->
  <section>
    <h2>⑩ 品目×月別 詳細FTE（全データ）</h2>
    {table_detail}
  </section>

</div>

<footer>
  FTE算出システム ／ 生成: {generated_at}
</footer>

</body>
</html>
"""


def _kpi_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f'</div>'
    )


class FY2029HTMLReporter:
    """
    FY2029 FTE算出結果をHTMLレポートとして出力するクラス。

    使い方:
        reporter = FY2029HTMLReporter(output_dir="output")
        reporter.generate(
            fte_df=fte_df,
            summary_df=summary_df,
            allocation_df=allocation_df,
            total_fte_df=total_fte_df,
            frequency_mode="lifecycle_adjusted",
        )
    """

    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        fte_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        allocation_df: pd.DataFrame,
        total_fte_df: pd.DataFrame,
        frequency_mode: str = "lifecycle_adjusted",
        filename: str = "fy2029_fte_report.html",
        total_fte_fy_df: Optional[pd.DataFrame] = None,
        optimal_fte_df: Optional[pd.DataFrame] = None,
        raw_summary_fy: Optional[pd.DataFrame] = None,
        per_launch_allocations: Optional[Dict[str, pd.DataFrame]] = None,
        digital_act_df: Optional[pd.DataFrame] = None,
    ) -> Path:
        """
        HTMLレポートを生成してファイルに保存する。

        Returns
        -------
        Path: 保存先のファイルパス
        """
        fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"

        # ---- KPI 計算 ----
        from fy2029_fte_calculator import CURRENT_MR_COUNT

        def _area_avg_fte(area: str) -> float:
            sub = fte_df[fte_df["area"] == area]
            if sub.empty:
                return 0.0
            return sub[fte_col].groupby(sub["month"]).sum().mean()

        total_fte_cs = _area_avg_fte("CS")
        total_fte_ps = _area_avg_fte("PS")
        gap_cs = total_fte_cs - CURRENT_MR_COUNT["CS"]
        gap_ps = total_fte_ps - CURRENT_MR_COUNT["PS"]

        avg_mr_ratio = summary_df["avg_mr_ratio"].mean()
        n_products = fte_df["product_id"].nunique()

        kpi_cards = "".join([
            _kpi_card("算出品目数", str(n_products), "CS+PS"),
            _kpi_card("CS 平均必要FTE", f"{total_fte_cs:.1f}", f"現行: {CURRENT_MR_COUNT['CS']}名"),
            _kpi_card("PS 平均必要FTE", f"{total_fte_ps:.1f}", f"現行: {CURRENT_MR_COUNT['PS']}名"),
            _kpi_card(
                "CS FTE過不足",
                f"{gap_cs:+.1f}",
                "プラス=不足 / マイナス=余剰",
            ),
            _kpi_card(
                "PS FTE過不足",
                f"{gap_ps:+.1f}",
                "プラス=不足 / マイナス=余剰",
            ),
            _kpi_card("平均MR比率", f"{avg_mr_ratio*100:.0f}%", "全品目・全月平均"),
        ])

        # ---- FC/SC サマリーテーブル ----
        fc_sc_summary = (
            fte_df.groupby("product_id")
            .agg(
                avg_fc=("fc_doctors", "mean"),
                avg_sc=("sc_doctors", "mean"),
                avg_total=("target_doctors", "mean"),
            )
            .reset_index()
            .round(0)
        )
        fc_sc_summary["SC比率"] = (
            fc_sc_summary["avg_sc"] / fc_sc_summary["avg_total"].replace(0, np.nan)
        ).fillna(0).round(3)
        fc_sc_summary.columns = ["品目", "FC医師数(平均)", "SC医師数(平均)",
                                  "合計ターゲット(平均)", "SC比率"]

        # ---- 詳細テーブル用カラム整理 ----
        detail_cols = [
            "product_id", "month", "area",
            "fc_doctors", "sc_doctors", "visit_frequency",
            "required_calls", fte_col,
            "mr_ratio", "digital_ratio", "mr_fte",
        ]
        col_labels = [
            "品目", "月", "領域",
            "FC医師数", "SC医師数", "訪問頻度",
            "必要コール数", "必要FTE",
            "MR比率", "Digital比率", "MR_FTE",
        ]
        available_cols = [c for c in detail_cols if c in fte_df.columns]
        detail_df = fte_df[available_cols].copy()
        detail_df.columns = col_labels[:len(available_cols)]

        # ---- 総FTEテーブル ----
        total_fte_display = total_fte_df.copy().rename(columns={
            "area": "領域", "month": "月",
            "total_fte": "合計必要FTE", "fiscal_year": "年度",
            "current_mr": "現行MR数", "fte_gap": "FTE過不足",
        })

        # ---- FYトレンド用チャート ----
        _fy_area_df = total_fte_fy_df if total_fte_fy_df is not None else pd.DataFrame()
        # ③セクション: FY2029のsummaryを使用（なければ最終FY）
        TARGET_FY = "FY2029"
        fy_list = sorted(summary_df["fiscal_year"].unique()) if "fiscal_year" in summary_df.columns else []
        if TARGET_FY not in fy_list and fy_list:
            TARGET_FY = fy_list[-1]
        summary_latest = (
            summary_df[summary_df["fiscal_year"] == TARGET_FY]
            if TARGET_FY in fy_list else summary_df
        )
        fy_range_display = (
            f"{fy_list[0]}〜{fy_list[-1]}" if len(fy_list) >= 2
            else (fy_list[0] if fy_list else TARGET_FY)
        )

        # ---- FYトレンドテーブル（品目×FY横持ち） ----
        if "fiscal_year" in summary_df.columns:
            pivot_fy = summary_df.pivot_table(
                index="product_id", columns="fiscal_year",
                values="avg_required_fte",
            ).reset_index().round(1)
            pivot_fy.columns.name = None
        else:
            pivot_fy = summary_df[["product_id", "avg_required_fte"]].round(1)

        # ---- デジタル活動サマリーテーブル ----
        _digital_df = digital_act_df if digital_act_df is not None else pd.DataFrame()
        if not _digital_df.empty and "product_id" in _digital_df.columns:
            _dg = _digital_df.copy()
            _dg_s = (
                _dg.groupby(["product_id", "activity_type"])
                .size()
                .reset_index(name="視聴数")
                .rename(columns={"product_id": "品目", "activity_type": "活動種別"})
            )
            # 月次平均視聴数を追加
            if "activity_date" in _dg.columns:
                _dg["activity_ym"] = _dg["activity_date"].str[:7]
            if "activity_ym" in _dg.columns:
                _n_months = _dg["activity_ym"].nunique()
                _dg_s["月平均視聴数"] = (_dg_s["視聴数"] / max(_n_months, 1)).round(1)
            digital_summary_html = _df_to_html(_dg_s, title="デジタル活動サマリー（全期間・視聴ログ）",
                                               bar_cols=["視聴数"])
        else:
            digital_summary_html = "<p style='color:#999;'>デジタル活動データなし</p>"

        # ---- HTMLレンダリング ----
        _opt_df = optimal_fte_df if optimal_fte_df is not None else pd.DataFrame()

        html = HTML_TEMPLATE.format(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            frequency_mode=frequency_mode,
            kpi_cards=kpi_cards,
            chart_fy_trend=_fig_to_html(fig_fy_trend(summary_df), "chart_fy_trend"),
            chart_fy_area=_fig_to_html(fig_fy_trend_area(_fy_area_df), "chart_fy_area"),
            table_fy_summary=_df_to_html(pivot_fy, title="品目×年度 平均FTE（横持ち）",
                                          bar_cols=[c for c in pivot_fy.columns if "FY" in str(c)]),
            chart_sim_vs_optimal=_fig_to_html(
                fig_sim_vs_optimal(summary_df, _opt_df), "chart_sim_vs_optimal"
            ),
            chart_fte_bar=_fig_to_html(fig_fte_by_product_bar(summary_latest, TARGET_FY), "chart_fte_bar"),
            chart_mr_digital=_fig_to_html(fig_mr_digital_ratio(summary_latest, TARGET_FY), "chart_mr_digital"),
            chart_monthly=_fig_to_html(fig_monthly_fte_trend(fte_df), "chart_monthly"),
            chart_fc_sc=_fig_to_html(fig_fc_sc_breakdown(fte_df), "chart_fc_sc"),
            chart_per_launch=_fig_to_html(
                fig_per_launch_allocation(per_launch_allocations or {}), "chart_per_launch"
            ),
            chart_ove=_fig_to_html(fig_ove_allocation(allocation_df), "chart_ove"),
            chart_raw_fte=_fig_to_html(
                fig_raw_fte_by_fy(raw_summary_fy if raw_summary_fy is not None else pd.DataFrame()),
                "chart_raw_fte",
            ),
            chart_headcount=_fig_to_html(fig_fte_vs_headcount(total_fte_df), "chart_headcount"),
            table_summary=_df_to_html(
                summary_latest.drop(columns=["fiscal_year"], errors="ignore"),
                title=f"品目別年度FTEサマリー（{TARGET_FY}）",
                bar_cols=["avg_required_fte", "avg_mr_fte"],
            ),
            table_fc_sc=_df_to_html(fc_sc_summary, title="FC/SC医師数内訳"),
            table_ove=_df_to_html(allocation_df, title="ドナー品目別 削減FTE詳細",
                                   bar_cols=["fte_reduction"]),
            table_headcount=_df_to_html(total_fte_display, title="領域×月別 FTE過不足一覧"),
            chart_digital_activity=_fig_to_html(fig_digital_activity(_digital_df), "chart_digital_activity"),
            chart_digital_trend=_fig_to_html(fig_digital_trend(_digital_df), "chart_digital_trend"),
            table_digital_summary=digital_summary_html,
            table_detail=_df_to_html(detail_df, title="全品目×全月 詳細FTE"),
            current_cs=CURRENT_MR_COUNT["CS"],
            current_ps=CURRENT_MR_COUNT["PS"],
            target_fy=TARGET_FY,
            fy_range_display=fy_range_display,
        )

        out_path = self.output_dir / filename
        out_path.write_text(html, encoding="utf-8")
        print(f"[OK] レポート出力: {out_path.resolve()}")
        return out_path
