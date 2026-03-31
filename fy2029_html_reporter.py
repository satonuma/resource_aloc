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


def fig_digital_effectiveness(score_df: pd.DataFrame) -> go.Figure:
    """品目別デジタル有効性スコア 水平バー（MMM・SOC・ライフサイクル合成）"""
    if score_df is None or score_df.empty:
        return go.Figure()

    df = score_df.copy().sort_values("digital_score", ascending=True)

    # レベル別カラー
    color_map = {"高": "#2ca02c", "中": "#ff7f0e", "低": "#d62728"}
    colors = df["digital_level"].map(color_map).fillna("#aec7e8")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="デジタル有効性スコア",
        y=df["product_id"],
        x=df["digital_score"] * 100,
        orientation="h",
        marker_color=colors.tolist(),
        text=df.apply(
            lambda r: f"{r['digital_score']*100:.0f}% [{r['digital_level']}]", axis=1
        ),
        textposition="outside",
        customdata=df[["mmm_digital_fraction", "digital_soc_rate", "lifecycle_adj"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "スコア: %{x:.1f}%<br>"
            "MMM デジタル応答比: %{customdata[0]:.3f}<br>"
            "SOC デジタル感受性: %{customdata[1]:.3f}<br>"
            "ライフサイクル補正: %{customdata[2]:+.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=(
            "品目別 デジタルチャネル有効性スコア（FY2029時点）<br>"
            "<sup>①MMM デジタル応答比（W=50%）②SOC デジタル感受性/想起率（W=50%）＋ライフサイクル補正</sup>"
        ),
        xaxis_title="スコア（%）",
        xaxis=dict(range=[0, 110]),
        height=420,
        template="plotly_white",
        showlegend=False,
    )
    # スコアしきい値ライン
    fig.add_vline(x=55, line_dash="dot", line_color="#2ca02c",
                  annotation_text="高（55%）", annotation_position="top right")
    fig.add_vline(x=35, line_dash="dot", line_color="#ff7f0e",
                  annotation_text="中（35%）", annotation_position="top right")
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
    """品目別 FC比率 / 訪問コスト重み（fc_weight）の棒グラフ"""
    if "fc_ratio" not in fte_df.columns:
        # 旧形式データ互換: 空グラフを返す
        return go.Figure()

    avg = (
        fte_df.groupby("product_id")[["fc_ratio", "fc_weight"]]
        .mean()
        .reset_index()
        .sort_values("fc_ratio", ascending=False)
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="FC比率（主訪問割合）",
        x=avg["product_id"],
        y=(avg["fc_ratio"] * 100).round(1),
        marker_color="#1f77b4",
        text=(avg["fc_ratio"] * 100).round(0).astype(int).astype(str) + "%",
        textposition="inside",
    ))
    fig.add_trace(go.Scatter(
        name="訪問コスト重み（fc_weight）",
        x=avg["product_id"],
        y=(avg["fc_weight"] * 100).round(1),
        mode="markers+lines",
        marker=dict(size=8, color="#d62728"),
        line=dict(dash="dot", color="#d62728"),
        yaxis="y2",
    ))

    fig.update_layout(
        title="品目別 FC比率 と 訪問コスト重み",
        xaxis_title="品目",
        yaxis=dict(title="FC比率 (%)", range=[0, 105]),
        yaxis2=dict(title="fc_weight (%)", overlaying="y", side="right", range=[0, 105]),
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
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">FC / SC コスト重み</div>
        <div style="font-size:12px;opacity:0.82;">品目ごとに fc_weight を設定。fc_weight = fc_ratio + (1−fc_ratio)×0.1</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 3</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">訪問頻度の推定</div>
        <div style="font-size:12px;opacity:0.82;">実績ベース頻度にライフサイクル補正（発売初期×1.3 / LOE前後×0.7→0.3）を適用</div>
      </div>
      <div style="background:rgba(255,255,255,0.12);border-radius:8px;padding:14px 16px;">
        <div style="font-size:11px;font-weight:700;letter-spacing:1px;opacity:0.7;margin-bottom:6px;">STEP 4</div>
        <div style="font-size:13px;font-weight:700;margin-bottom:4px;">デジタル有効性スコア</div>
        <div style="font-size:12px;opacity:0.82;">MR FTEとは独立して算出。①MMM デジタル応答比②SOC デジタル感受性③ライフサイクル補正で品目別チャネル戦略示唆を提供</div>
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
      <div>{chart_digital_score}</div>
    </div>
    {table_summary}
    <h3 style="margin-top:24px;font-size:15px;color:#444;">デジタルチャネル有効性スコアについて</h3>
    <p style="font-size:13px;color:#555;line-height:1.7;margin-bottom:12px;">
      MR FTE（380名CS / 45名PS）はMR活動のみから直接算出しており、デジタルチャネルはFTE計算に影響しません。<br>
      デジタル有効性スコアはチャネル戦略上の示唆として独立して算出します。<br>
      <strong>①MMM デジタル応答比（W=50%）</strong>：Hill関数の総効果量 dig_val/(mr_val+dig_val)。活動数補正済みのデジタルチャネル効果量。MRチャネル: 面談/面談_アポ/説明会、デジタルチャネル: Web講演会/Webinar/e-contents/メール（slope_m=1.0固定: ミカエリス-メンテン型）。<br>
      <strong>②SOC デジタル感受性（W=50%）</strong>：digital_soc_rate（下表）を直接使用。医師が視聴1回でどれだけ想起するかの確率（0〜1）。<br>
      <strong>ライフサイクル補正</strong>：LOE後 −25pt / LOE1年未満 −20pt / LOE1〜3年 −10pt、発売1年未満 −5pt、成長期（1〜3年） +8pt、成熟期 +5pt。<br>
      <strong>スコア解釈</strong>：高（55%以上）= m3等デジタル積極活用余地あり / 中（35〜55%）= 選択的活用 / 低（35%未満）= MR中心維持
    </p>
    {table_soc_params}
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

  <!-- Section MMM: レスポンスカーブ & パラメータ調整 -->
  <section>
    <h2>MMM レスポンスカーブ & ライフサイクル調整</h2>
    {mmm_section}
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


def generate_logic_document(output_dir: str = "output") -> Path:
    """FTE算出ロジックドキュメントを生成してHTMLファイルに書き出す。

    Args:
        output_dir: 出力先ディレクトリ（デフォルト: "output"）

    Returns:
        出力ファイルのPathオブジェクト
    """
    from datetime import datetime as _dt

    generated_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    html = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MR FTE算出システム ロジックドキュメント (FY2026-2035)</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --navy: #0d1b2a;
    --navy-mid: #132336;
    --navy-dark: #0f2540;
    --blue: #2563eb;
    --blue-light: #3b82f6;
    --blue-pale: #dbeafe;
    --amber: #f59e0b;
    --amber-pale: #fef3c7;
    --green: #10b981;
    --green-pale: #d1fae5;
    --text: #1e293b;
    --text-muted: #64748b;
    --bg: #f1f5f9;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.06);
    --shadow-hover: 0 4px 12px rgba(0,0,0,0.12), 0 8px 32px rgba(0,0,0,0.08);
  }

  body {
    font-family: "Inter", "Segoe UI", "Hiragino Sans", "Noto Sans JP", sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    min-height: 100vh;
    line-height: 1.6;
    font-size: 15px;
  }

  /* ── Sidebar ─────────────────────────── */
  nav#sidebar {
    width: 248px;
    min-width: 248px;
    background: var(--navy);
    color: #c9d6df;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    padding: 0 0 32px;
    flex-shrink: 0;
    scrollbar-width: thin;
    scrollbar-color: #2d4a6a transparent;
  }
  nav#sidebar::-webkit-scrollbar { width: 4px; }
  nav#sidebar::-webkit-scrollbar-thumb { background: #2d4a6a; border-radius: 2px; }

  .sidebar-logo {
    padding: 22px 20px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px;
  }
  .sidebar-logo .logo-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--blue-light);
    margin-bottom: 2px;
  }
  .sidebar-logo .logo-sub {
    font-size: 11px;
    color: #7a99b0;
  }

  nav#sidebar a {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 20px;
    color: #8fa8be;
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    border-left: 3px solid transparent;
    transition: all 0.18s ease;
  }
  nav#sidebar a:hover {
    color: #e2eaf0;
    background: rgba(255,255,255,0.06);
    border-left-color: var(--blue-light);
  }
  nav#sidebar a.active {
    color: #ffffff;
    background: rgba(37,99,235,0.18);
    border-left-color: var(--blue);
  }
  .nav-step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(37,99,235,0.25);
    color: var(--blue-light);
    font-size: 10px;
    font-weight: 700;
    flex-shrink: 0;
  }
  nav#sidebar .nav-section-label {
    padding: 14px 20px 4px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a6b82;
  }

  /* ── Main content ───────────────────── */
  main {
    flex: 1;
    overflow-y: auto;
    padding: 0 0 64px;
    max-width: 960px;
  }

  /* ── Hero ───────────────────────────── */
  .hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #132d50 60%, #1a3a6b 100%);
    color: white;
    padding: 56px 48px 48px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(37,99,235,0.15) 0%, transparent 70%);
  }
  .hero::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 40%;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
  }
  .hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--blue-light);
    margin-bottom: 12px;
    position: relative;
  }
  .hero h1 {
    font-size: 32px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 12px;
    position: relative;
  }
  .hero h1 span { color: var(--blue-light); }
  .hero-desc {
    font-size: 15px;
    color: #94afc5;
    max-width: 560px;
    margin-bottom: 36px;
    position: relative;
  }
  .kpi-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    position: relative;
  }
  .kpi-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 8px;
    padding: 12px 18px;
    backdrop-filter: blur(8px);
    transition: background 0.2s;
  }
  .kpi-badge:hover { background: rgba(255,255,255,0.13); }
  .kpi-badge .kpi-value {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin-bottom: 4px;
  }
  .kpi-badge .kpi-label {
    font-size: 11px;
    color: #7aa0be;
    font-weight: 500;
  }

  /* ── Section cards ──────────────────── */
  .content-wrap { padding: 32px 40px; }

  .section-card {
    background: var(--card-bg);
    border-radius: 12px;
    box-shadow: var(--shadow);
    margin-bottom: 28px;
    overflow: hidden;
    transition: box-shadow 0.2s ease;
    border: 1px solid var(--border);
  }
  .section-card:hover { box-shadow: var(--shadow-hover); }

  .section-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 28px 18px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  }
  .step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: var(--blue);
    color: white;
    font-size: 14px;
    font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(37,99,235,0.35);
  }
  .step-badge.star {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    box-shadow: 0 2px 8px rgba(245,158,11,0.35);
  }
  .step-badge.digital {
    background: linear-gradient(135deg, #0891b2, #0e7490);
    box-shadow: 0 2px 8px rgba(8,145,178,0.35);
  }
  .section-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
  }
  .section-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 2px;
  }
  .section-body { padding: 24px 28px; }

  /* ── Formula boxes ──────────────────── */
  .formula-box {
    background: var(--navy-dark);
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
    font-family: "Cascadia Code", "Fira Code", "Consolas", "Courier New", monospace;
    font-size: 13.5px;
    line-height: 1.85;
    color: #e2f0ff;
    border-left: 4px solid var(--blue);
    overflow-x: auto;
  }
  .formula-box .formula-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--blue-light);
    margin-bottom: 8px;
    font-family: "Inter", "Segoe UI", sans-serif;
  }
  .formula-box .highlight { color: #fbbf24; font-weight: 600; }
  .formula-box .comment { color: #64748b; }

  .formula-box-compact {
    background: var(--navy-dark);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 10px 0;
    font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
    font-size: 13px;
    line-height: 1.75;
    color: #e2f0ff;
    border-left: 3px solid var(--blue-light);
    overflow-x: auto;
  }

  /* ── Summary formula box (overview) ── */
  .summary-formula-box {
    background: var(--navy-dark);
    border-radius: 10px;
    padding: 24px 28px;
    margin: 20px 0;
    font-family: "Cascadia Code", "Fira Code", "Consolas", monospace;
    font-size: 12.5px;
    line-height: 2.0;
    color: #d4e8ff;
    border: 1px solid rgba(37,99,235,0.3);
    overflow-x: auto;
  }
  .summary-formula-box .sfb-section {
    color: #60a5fa;
    font-weight: 700;
    font-size: 11.5px;
    letter-spacing: 0.05em;
    margin-top: 14px;
    margin-bottom: 4px;
    display: block;
    font-family: "Inter", "Segoe UI", sans-serif;
    text-transform: uppercase;
  }
  .summary-formula-box .sfb-section:first-child { margin-top: 0; }

  /* ── Pipeline flow ──────────────────── */
  .pipeline-wrap {
    padding: 8px 0 16px;
    overflow-x: auto;
  }
  .pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: nowrap;
    min-width: max-content;
  }
  .pipeline-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }
  .pipeline-box {
    background: var(--blue);
    color: white;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 600;
    text-align: center;
    white-space: nowrap;
    min-width: 90px;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3);
    transition: transform 0.15s, box-shadow 0.15s;
  }
  .pipeline-box:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(37,99,235,0.4);
  }
  .pipeline-box.input-box {
    background: linear-gradient(135deg, #374151, #1f2937);
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
  }
  .pipeline-box.output-box {
    background: linear-gradient(135deg, #059669, #047857);
    box-shadow: 0 2px 8px rgba(5,150,105,0.3);
  }
  .pipeline-box.digital-box {
    background: linear-gradient(135deg, #0891b2, #0e7490);
    box-shadow: 0 2px 8px rgba(8,145,178,0.3);
  }
  .pipeline-step-num {
    font-size: 10px;
    color: rgba(255,255,255,0.7);
    font-weight: 500;
  }
  .pipeline-arrow {
    color: #94a3b8;
    font-size: 20px;
    padding: 0 6px;
    line-height: 1;
    margin-bottom: 18px;
  }
  .pipeline-branch {
    display: flex;
    align-items: center;
    margin-top: 16px;
    gap: 12px;
  }
  .branch-label {
    font-size: 11px;
    color: var(--text-muted);
    font-style: italic;
  }

  /* ── Info boxes ─────────────────────── */
  .info-box {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 14px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
  }
  .info-box.blue {
    background: var(--blue-pale);
    border-left: 4px solid var(--blue);
    color: #1e40af;
  }
  .info-box.amber {
    background: var(--amber-pale);
    border-left: 4px solid var(--amber);
    color: #92400e;
  }
  .info-box.green {
    background: var(--green-pale);
    border-left: 4px solid var(--green);
    color: #065f46;
  }
  .info-icon { font-size: 18px; flex-shrink: 0; }

  /* ── Amber number badge ─────────────── */
  .num-badge {
    display: inline-block;
    background: var(--amber);
    color: #1c0a00;
    border-radius: 5px;
    padding: 1px 8px;
    font-size: 13px;
    font-weight: 700;
    margin: 0 2px;
  }

  /* ── Tables ─────────────────────────── */
  .data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
    margin: 14px 0;
  }
  .data-table th {
    background: #f1f5f9;
    color: var(--text-muted);
    font-weight: 600;
    font-size: 11.5px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 2px solid var(--border);
    white-space: nowrap;
  }
  .data-table td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  .data-table tbody tr:hover { background: #f8fafc; }
  .data-table tbody tr:last-child td { border-bottom: none; }

  .tag {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
  }
  .tag-new { background: #fce7f3; color: #9d174d; }
  .tag-existing { background: var(--green-pale); color: #065f46; }
  .tag-cs { background: var(--blue-pale); color: #1e40af; }
  .tag-ps { background: #ede9fe; color: #5b21b6; }

  /* ── Two-column layout ───────────────── */
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 14px 0;
  }
  @media (max-width: 720px) { .two-col { grid-template-columns: 1fr; } }

  .mini-card {
    background: #f8fafc;
    border-radius: 8px;
    padding: 16px 18px;
    border: 1px solid var(--border);
  }
  .mini-card h4 {
    font-size: 13px;
    font-weight: 700;
    color: var(--text-muted);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* ── Generated at footer ────────────── */
  .doc-footer {
    text-align: center;
    padding: 32px;
    font-size: 12px;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    margin-top: 8px;
  }

  h3 {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    margin: 20px 0 10px;
  }
  h3:first-child { margin-top: 0; }
  p { margin: 8px 0; color: var(--text); }
  ul, ol { margin: 8px 0 8px 20px; }
  li { margin: 4px 0; }

  /* scrollbar for main */
  main::-webkit-scrollbar { width: 6px; }
  main::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
</head>
<body>

<!-- ═══════════════════════ SIDEBAR ═══════════════════════ -->
<nav id="sidebar">
  <div class="sidebar-logo">
    <div class="logo-title">MR FTE System</div>
    <div class="logo-sub">FY2026–2035 ロジック仕様書</div>
  </div>

  <div class="nav-section-label">概要</div>
  <a href="#overview" class="active">
    <span class="nav-step-badge">★</span> 概要・全体フロー
  </a>
  <a href="#data-files">
    <span class="nav-step-badge">D</span> データファイル一覧
  </a>

  <div class="nav-section-label">FTE算出ステップ</div>
  <a href="#step1">
    <span class="nav-step-badge">1</span> ターゲット医師数
  </a>
  <a href="#step2">
    <span class="nav-step-badge">2</span> FC/SC分割
  </a>
  <a href="#step3">
    <span class="nav-step-badge">3</span> 訪問頻度推定
  </a>
  <a href="#step4">
    <span class="nav-step-badge">4</span> MR FTE算出
  </a>
  <a href="#step5">
    <span class="nav-step-badge">5</span> HC正規化・離散化
  </a>
  <a href="#step6">
    <span class="nav-step-badge">6</span> 新発売品FTE配分
  </a>

  <div class="nav-section-label">独立分析</div>
  <a href="#digital">
    <span class="nav-step-badge">D</span> デジタル有効性スコア
  </a>
  <a href="#mmm-lifecycle">
    <span class="nav-step-badge">M</span> MMMパラメータ調整
  </a>
</nav>

<!-- ═══════════════════════ MAIN ═══════════════════════════ -->
<main>

<!-- ── HERO ──────────────────────────────────────────────── -->
<div class="hero">
  <div class="hero-eyebrow">Technical Specification Document · Ver 2.0</div>
  <h1>MR FTE 算出システム<br><span>ロジックドキュメント</span></h1>
  <p class="hero-desc">
    Marketing-Mix Model (MMM) および ライフサイクル調整を組み合わせた
    MR（医薬情報担当者）FTE シミュレーションシステムの完全技術仕様書。
    FY2026〜FY2035 の10年間、19品目を対象とした月次最適配分を定義します。
  </p>
  <div class="kpi-row">
    <div class="kpi-badge">
      <div class="kpi-value">FY2026–35</div>
      <div class="kpi-label">対象期間（10年間）</div>
    </div>
    <div class="kpi-badge">
      <div class="kpi-value">19品目</div>
      <div class="kpi-label">対象医薬品</div>
    </div>
    <div class="kpi-badge">
      <div class="kpi-value">CS 380 / PS 45</div>
      <div class="kpi-label">HC正規化ターゲット（名）</div>
    </div>
    <div class="kpi-badge">
      <div class="kpi-value">品目×月別</div>
      <div class="kpi-label">FTE算出粒度</div>
    </div>
  </div>
</div>

<div class="content-wrap">

<!-- ── OVERVIEW ──────────────────────────────────────────── -->
<div class="section-card" id="overview">
  <div class="section-header">
    <span class="step-badge star">★</span>
    <div>
      <div class="section-title">概要・全体フロー</div>
      <div class="section-subtitle">システム全体の処理パイプラインと主要数式のサマリー</div>
    </div>
  </div>
  <div class="section-body">

    <h3>処理パイプライン</h3>
    <div class="pipeline-wrap">
      <div class="pipeline">
        <div class="pipeline-node">
          <div class="pipeline-box input-box">データ入力</div>
          <div class="pipeline-step-num">CSV/設定</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box">ターゲット医師数</div>
          <div class="pipeline-step-num">STEP 1</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box">FC/SC訪問コスト重み</div>
          <div class="pipeline-step-num">STEP 2</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box">訪問頻度</div>
          <div class="pipeline-step-num">STEP 3</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box">FTE算出</div>
          <div class="pipeline-step-num">STEP 4</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box">正規化</div>
          <div class="pipeline-step-num">STEP 5</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box">新製品配分</div>
          <div class="pipeline-step-num">STEP 6</div>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
          <div class="pipeline-box output-box">MR FTE 出力</div>
          <div class="pipeline-step-num">最終結果</div>
        </div>
      </div>
      <div class="pipeline-branch">
        <div class="branch-label">独立分析：</div>
        <div class="pipeline-box digital-box" style="min-width:200px; font-size:12px;">
          デジタル有効性スコア（独立分析）
        </div>
      </div>
    </div>

    <h3 style="margin-top:28px;">主要式まとめ</h3>
    <div class="summary-formula-box">
<span class="sfb-section">STEP 4 — MR FTE算出</span>FTE = (R医師×r_freq + W医師×w_freq) × lc_adj × fc_weight ÷ (実稼働日数 × コール/日)
  CS: コール/日 = 2.5  │  PS: コール/日 = 1.5
  実稼働日数: working_days.csv より年月別に設定（18〜23日/月、土日祝除外）

<span class="sfb-section">STEP 5 — ヘッドカウント正規化</span>adjusted_FTE_i = base_FTE_i × (HC_target ÷ Σ base_FTE)
  CS目標: 380名   PS目標: 45名

<span class="sfb-section">STEP 6 — 新製品 ドナー選定（限界ROI）</span>marginal_ROI(p) = β_m(p) × ec_m(p) ÷ (ec_m(p) + x)²
  ※ slope_m = 1 固定（ミカエリス-メンテン型）により簡略化
削減上限: fy2026_apr_fte(p) × 0.5 を最低保護（50%まで削減可）

<span class="sfb-section">デジタル有効性スコア（FTEとは独立）</span>base  = 0.5 × [Σdig_channels hill(x_d) ÷ (Σmr_channels hill(x_m) + Σdig_channels hill(x_d))]
        + 0.5 × digital_soc_rate
score = clip(base + lifecycle_adj,  0.0,  1.0)
MRチャネル: 面談/面談_アポ/説明会  デジタルチャネル: Web講演会/Webinar/e-contents/メール
    </div>

    <h3>対象品目一覧（19品目）</h3>
    <table class="data-table">
      <thead>
        <tr>
          <th>#</th>
          <th>品目ID</th>
          <th>エリア</th>
          <th>区分</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>1</td><td>GLI</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>2</td><td>CUV</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>3</td><td>HYQ</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>4</td><td>INT</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>5</td><td>TRI</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>6</td><td>ENT</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>7</td><td>LIV</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>8</td><td>REV</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>9</td><td>ALC</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>10</td><td>VYV</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>11</td><td>VPR</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>12</td><td>GLO</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-new">新発売品</span></td></tr>
        <tr><td>13</td><td>OVE</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-new">新発売品</span></td></tr>
        <tr><td>14</td><td>Zaso</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-new">新発売品</span></td></tr>
        <tr><td>15</td><td>WSA</td><td><span class="tag tag-cs">CS</span></td><td><span class="tag tag-new">新発売品</span></td></tr>
        <tr><td>16</td><td>LVM</td><td><span class="tag tag-ps">PS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>17</td><td>TKZ</td><td><span class="tag tag-ps">PS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>18</td><td>RPL</td><td><span class="tag tag-ps">PS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
        <tr><td>19</td><td>VON</td><td><span class="tag tag-ps">PS</span></td><td><span class="tag tag-existing">既存品</span></td></tr>
      </tbody>
    </table>
  </div>
</div><!-- /overview -->

<!-- ── STEP 1 ─────────────────────────────────────────────── -->
<div class="section-card" id="step1">
  <div class="section-header">
    <span class="step-badge">1</span>
    <div>
      <div class="section-title">ターゲット医師数の決定</div>
      <div class="section-subtitle">R/W医師ティアによるCSV直接管理・年度別増減対応</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      品目ごとに患者数ポテンシャルに基づき <strong>R医師</strong>（患者数かなり多め・高頻度訪問対象）と
      <strong>W医師</strong>（患者数多め・標準頻度訪問対象）に分類し、CSV で直接管理します。
      年度別の増減（発売後の浸透・LOE後の活動縮小）も <code>target_doctor_yearly.csv</code> で指定可能です。
    </p>
    <h3>主要式</h3>
    <div class="formula-box">
      <div class="formula-label">STEP 1 — ターゲット医師数</div>
target_drs(p) = r_doctors(p) + w_doctors(p)
<br><small style="opacity:0.75">※ target_doctor_yearly.csv に年度別指定がある場合はその値を優先使用</small>
    </div>
    <h3>医師ティア区分</h3>
    <table class="data-table">
      <thead>
        <tr>
          <th>ティア</th>
          <th>定義</th>
          <th>標準訪問頻度</th>
          <th>データソース</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><strong>R医師</strong></td><td>患者数かなり多め：高ポテンシャル層</td><td><span class="num-badge">2.0回/月</span></td><td>target_doctors.csv</td></tr>
        <tr><td><strong>W医師</strong></td><td>患者数多め：標準ポテンシャル層</td><td><span class="num-badge">1.0回/月</span></td><td>target_doctors.csv</td></tr>
      </tbody>
    </table>
    <div class="info-box blue">
      <span class="info-icon">ℹ️</span>
      <span>
        訪問頻度はライフサイクル調整係数を乗算：
        <code>effective_freq = tier_freq × lifecycle_adj(month)</code>。
        新製品は <code>target_doctor_yearly.csv</code> で年度別に医師数を増加指定（発売後の市場浸透を表現）。
      </span>
    </div>
  </div>
</div><!-- /step1 -->

<!-- ── STEP 2 ─────────────────────────────────────────────── -->
<div class="section-card" id="step2">
  <div class="section-header">
    <span class="step-badge">2</span>
    <div>
      <div class="section-title">FC / SC 訪問コスト重み</div>
      <div class="section-subtitle">品目単位の訪問コスト重み設定（医師分類ではない）</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      FC（First Call）とSC（Second Call）は<strong>医師の分類ではなく、品目の訪問種別</strong>です。
      ある医師Aに対してTRIはFC（主訪問）、INTはSC（ついで訪問）として同一訪問内で活動します。
      ターゲット医師はR/Wティアで管理し、FC/SCは訪問コストの重みとして品目ごとに設定します。
    </p>
    <h3>分割式</h3>
    <div class="formula-box">
      <div class="formula-label">STEP 2 — FC/SC 訪問コスト重み</div>
fc_weight(p) = fc_ratio(p) + (1 − fc_ratio(p)) × SC_COEFFICIENT
required_calls(p) = raw_calls(p) × fc_weight(p)
<br><small style="opacity:0.75">raw_calls = R医師×r_freq×lc_adj + W医師×w_freq×lc_adj</small>

<span class="comment"># オーバーラップ（複数品目で同一医師を訪問）を考慮したFCオフセット</span>
effective_FC(p) = FC_drs(p) − overlap_offset(p)
    </div>
  </div>
</div><!-- /step2 -->

<!-- ── STEP 3 ─────────────────────────────────────────────── -->
<div class="section-card" id="step3">
  <div class="section-header">
    <span class="step-badge">3</span>
    <div>
      <div class="section-title">訪問頻度の推定</div>
      <div class="section-subtitle">製品ライフサイクルと競合環境に基づく月次訪問頻度の算出</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      FC医師あたりの月間訪問頻度は、製品の <strong>ライフサイクルステージ</strong>・
      <strong>競合激しさ</strong>・<strong>プロモーション優先度</strong> の3要素で決まります。
    </p>
    <div class="formula-box">
      <div class="formula-label">STEP 3 — 訪問頻度（FC）</div>
effective_freq(p, tier, t) = target_freq(p, tier, FY) × achievement_rate(p, tier, FY)
  ※ visit_freq.csv で品目×年度×R/Wティア別に管理（target_freq, achievement_rate）
  ※ visit_freq.csv に該当データがある場合 → lc_adj は適用しない（二重適用防止）
  ※ visit_freq.csv に該当データがない場合 → lc_adj（ライフサイクル係数）で補正
    </div>
    <h3>ライフサイクル係数テーブル</h3>
    <table class="data-table">
      <thead>
        <tr>
          <th>ライフサイクルステージ</th>
          <th>説明</th>
          <th>係数（lifecycle_factor）</th>
        </tr>
      </thead>
      <tbody>
        <tr><td><strong>Launch</strong></td><td>発売から12ヶ月以内</td><td><span class="num-badge">1.5</span></td></tr>
        <tr><td><strong>Early Growth</strong></td><td>発売13〜36ヶ月</td><td><span class="num-badge">1.2</span></td></tr>
        <tr><td><strong>Mature</strong></td><td>安定期</td><td><span class="num-badge">1.0</span></td></tr>
        <tr><td><strong>Late Growth</strong></td><td>成熟後期・微減期</td><td><span class="num-badge">0.85</span></td></tr>
        <tr><td><strong>Decline</strong></td><td>特許切れ・後発品参入後</td><td><span class="num-badge">0.6</span></td></tr>
      </tbody>
    </table>
    <div class="info-box green">
      <span class="info-icon">✅</span>
      <span>
        SC医師への訪問頻度は固定の係数 <span class="num-badge">0.1</span> を乗じた値を使用します（FTE計算内で処理）。
      </span>
    </div>
  </div>
</div><!-- /step3 -->

<!-- ── STEP 4 ─────────────────────────────────────────────── -->
<div class="section-card" id="step4">
  <div class="section-header">
    <span class="step-badge">4</span>
    <div>
      <div class="section-title">MR FTE 算出</div>
      <div class="section-subtitle">コール数とMRキャパシティから品目×月別FTEを計算</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      R医師数・W医師数・訪問頻度（visit_freq.csv）から <strong>月間総コール数</strong> を求め、
      MR1名あたりのキャパシティで割ることで品目別のFTEを算出します。
      デジタルチャネルの比率は <strong>FTE計算には含めません</strong>（独立スコアとして別途算出）。
    </p>
    <div class="formula-box">
      <div class="formula-label">STEP 4 — MR FTE 算出式（コア）</div>
FTE(p, t) = (R_drs × r_freq + W_drs × w_freq) × lc_adj × fc_weight(p)
            ÷ (実稼働日数 × コール/日)

<span class="highlight">CS エリア:</span>  コール/日 = <span class="highlight">2.5</span>  →  月間キャパ = <span class="highlight">50</span> コール/FTE
<span class="highlight">PS エリア:</span>  コール/日 = <span class="highlight">1.5</span>  →  月間キャパ = <span class="highlight">30</span> コール/FTE
    </div>
    <div class="two-col">
      <div class="mini-card">
        <h4>CS — コールキャパシティ</h4>
        <p>稼働日数： <span class="num-badge">working_days.csv</span> <span style="font-size:12px;color:#666;">（18〜23日/月、土日祝日除外）</span></p>
        <p>コール/日： <span class="num-badge">2.5</span></p>
        <p>月間キャパ： <span class="num-badge">50コール/FTE</span></p>
      </div>
      <div class="mini-card">
        <h4>PS — コールキャパシティ</h4>
        <p>稼働日数： <span class="num-badge">working_days.csv</span> <span style="font-size:12px;color:#666;">（18〜23日/月、土日祝日除外）</span></p>
        <p>コール/日： <span class="num-badge">1.5</span></p>
        <p>月間キャパ： <span class="num-badge">30コール/FTE</span></p>
      </div>
    </div>
    <div class="info-box amber">
      <span class="info-icon">⚠️</span>
      <span>
        SC医師へのコール数は <code>SC_drs × 0.1</code> として計上します（SC医師は低頻度訪問の想定）。
        このSC係数 <strong>0.1</strong> はシステム固定値です。
      </span>
    </div>
  </div>
</div><!-- /step4 -->

<!-- ── STEP 5 ─────────────────────────────────────────────── -->
<div class="section-card" id="step5">
  <div class="section-header">
    <span class="step-badge">5</span>
    <div>
      <div class="section-title">HC 正規化・離散化</div>
      <div class="section-subtitle">エリア合計FTEをヘッドカウント目標値に正規化</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      STEP4で算出した品目別FTEの合計は、そのままではHC目標（CS:380名、PS:45名）と
      一致しません。比例スケーリングによって正規化し、整数化（離散化）します。
    </p>
    <h3>正規化式</h3>
    <div class="formula-box">
      <div class="formula-label">STEP 5 — HC正規化</div>
adjusted_FTE_i = base_FTE_i × (HC_target ÷ Σ<sub>i</sub> base_FTE_i)

<span class="highlight">CS:</span>  HC_target = <span class="highlight">380</span>名
<span class="highlight">PS:</span>  HC_target = <span class="highlight">45</span>名
    </div>
    <h3>H1 / H2 ステップ処理</h3>
    <div class="info-box blue">
      <span class="info-icon">ℹ️</span>
      <span>
        正規化は <strong>H1（4〜9月）</strong> と <strong>H2（10〜3月）</strong> の半期単位で実施します。
        半期ごとに合計HCが目標値に一致するよう調整することで、中途採用・退職の影響を吸収します。
      </span>
    </div>
    <div class="info-box amber">
      <span class="info-icon">⚠️</span>
      <span>
        離散化（整数化）には <strong>最大剰余法（Largest Remainder Method）</strong> を使用します。
        丸め誤差の合計が目標HCと完全に一致することを保証します。
      </span>
    </div>
  </div>
</div><!-- /step5 -->

<!-- ── STEP 6 ─────────────────────────────────────────────── -->
<div class="section-card" id="step6">
  <div class="section-header">
    <span class="step-badge">6</span>
    <div>
      <div class="section-title">新発売品 FTE 配分</div>
      <div class="section-subtitle">限界ROI最小化によるドナー品目選定と新製品へのFTE移転</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      新発売品（is_new=True）へ必要なFTEを確保するため、
      既存品から <strong>限界ROIが最も低いもの</strong> を優先的にドナーとしてFTEを削減します。
      FY2026年4月時点のFTEを削減下限の基準（<code>fy2026_apr_fte</code>）として保護します。
    </p>
    <h3>限界ROI式（MMM hill関数ベース）</h3>
    <div class="formula-box">
      <div class="formula-label">STEP 6 — 限界ROI（ドナー選定基準）</div>
marginal_ROI(p, x) = β_m(p) × ec_m(p) ÷ (ec_m(p) + x)²
  ※ slope_m = 1 固定（ミカエリス-メンテン型）の場合の簡略形
  ※ 活動量 x が増えるほど限界ROIは低下（逓減）
  ※ β_m はチャネル別MMMパラメータの合算値

<span class="comment"># パラメータ定義</span>
β_m(p)  : 製品pのMMMスケール係数（最大効果量）
slope_m : 1.0 固定（ミカエリス-メンテン型）
ec_m(p) : 半最大効果濃度（x軸: FTE換算）
x       : 現在のFTE投入量
    </div>
    <h3>fy2026_apr_fte による最低保護</h3>
    <div class="formula-box-compact">
min_FTE(p) = fy2026_apr_fte(p) × 0.5
<span style="color:#94a3b8;"># 各品目は FY2026年4月FTEの50%以上を保護（最大削減率 50%）</span>
    </div>
    <h3>ドナー選定アルゴリズム</h3>
    <ol style="margin-left:20px; line-height:2.0;">
      <li>全既存品の現FTEにおける <strong>限界ROI</strong> を計算</li>
      <li>限界ROI昇順にソート（ROIが低い＝削減しても損失が少ない）</li>
      <li>必要FTE充足まで、削減上限（<code>min_FTE</code>）を超えない範囲でドナーからFTEを取得</li>
      <li>取得したFTEを新発売品の <code>launch_fte_requirement</code> に充当</li>
    </ol>
    <div class="info-box green">
      <span class="info-icon">✅</span>
      <span>
        <code>fy2026_apr_fte</code> は「FY2026年4月の正規化済みFTE」をキャッシュした値であり、
        STEP5完了後に確定します。以降の全年度で削減下限の基準として参照されます。
      </span>
    </div>
  </div>
</div><!-- /step6 -->

<!-- ── DIGITAL ──────────────────────────────────────────────── -->
<div class="section-card" id="digital">
  <div class="section-header">
    <span class="step-badge digital">D</span>
    <div>
      <div class="section-title">デジタル有効性スコア</div>
      <div class="section-subtitle">MR FTEとは独立したデジタルチャネルの有効性推定</div>
    </div>
  </div>
  <div class="section-body">
    <p>
      デジタルチャネル（Web面談・eDetail・メール等）の有効性を
      <strong>0.0〜1.0</strong> のスコアで表現します。
      このスコアは <strong>MR FTE算出には影響しません</strong>（完全独立の分析）。
    </p>
    <h3>スコア算出式</h3>
    <div class="formula-box">
      <div class="formula-label">デジタル有効性スコア — 算出式</div>
<span class="highlight">Component A</span> (チャネルシェア、ウェイト 50%):
  comp_A = dig_hill(x_d) ÷ (mr_hill(x_m) + dig_hill(x_d))

<span class="highlight">Component B</span> (SOC デジタル率、ウェイト 50%):
  comp_B = digital_soc_rate

<span class="highlight">ベーススコア:</span>
  base = 0.5 × comp_A + 0.5 × comp_B

<span class="highlight">最終スコア（ライフサイクル調整後）:</span>
  score = clip(base + lifecycle_adj,  0.0,  1.0)
    </div>
    <h3>3コンポーネント分解</h3>
    <div class="two-col">
      <div class="mini-card">
        <h4>① チャネルシェア (50%)</h4>
        <p>デジタルHillとMR Hillの比率で、デジタルが占めるコール効果シェアを推定</p>
        <p style="margin-top:8px; font-size:12px; color:var(--text-muted);">
          <code>dig_hill</code>、<code>mr_hill</code> はそれぞれMMMから推定されたHill関数
        </p>
      </div>
      <div class="mini-card">
        <h4>② デジタルSOC率 (50%)</h4>
        <p>業界全体の Share of Channel（デジタル接触比率）を外部データから取得</p>
        <p style="margin-top:8px; font-size:12px; color:var(--text-muted);">
          年次更新値。製品・エリアによって異なる場合あり
        </p>
      </div>
    </div>
    <h3>ライフサイクル調整（lifecycle_adj）</h3>
    <table class="data-table">
      <thead>
        <tr>
          <th>ライフサイクルステージ</th>
          <th>lifecycle_adj</th>
          <th>理由</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Launch</td><td><span class="num-badge">−0.10</span></td><td>発売直後はMR対面が重要</td></tr>
        <tr><td>Early Growth</td><td><span class="num-badge">−0.05</span></td><td>関係構築フェーズ</td></tr>
        <tr><td>Mature</td><td><span class="num-badge">0.00</span></td><td>調整なし（ベースライン）</td></tr>
        <tr><td>Late Growth</td><td><span class="num-badge">+0.05</span></td><td>デジタルで効率化可能</td></tr>
        <tr><td>Decline</td><td><span class="num-badge">+0.10</span></td><td>MRより低コストなデジタルを優先</td></tr>
      </tbody>
    </table>
    <h3>スコアレベル解釈</h3>
    <table class="data-table">
      <thead>
        <tr><th>スコア範囲</th><th>レベル</th><th>推奨アクション</th></tr>
      </thead>
      <tbody>
        <tr><td>0.0 – 0.3</td><td>低</td><td>MR対面を維持、デジタルは補助的</td></tr>
        <tr><td>0.3 – 0.6</td><td>中</td><td>MRとデジタルのハイブリッド運用</td></tr>
        <tr><td>0.6 – 1.0</td><td>高</td><td>デジタル主導、MTR削減を検討</td></tr>
      </tbody>
    </table>
  </div>
</div><!-- /digital -->

<!-- ── MMM ライフサイクル調整 ─────────────────────────── -->
<div class="section-card" id="mmm-lifecycle">
  <div class="section-header">
    <span class="step-badge" style="background:var(--amber);">M</span>
    <div>
      <div class="section-title">MMMパラメータ ライフサイクル調整</div>
      <div class="section-subtitle">事後分布パラメータをLOE・新製品・競合参入・デジタルトレンドで将来補正</div>
    </div>
  </div>
  <div class="section-body">

    <h3>パラメータ一覧と性質</h3>
    <table class="data-table">
      <thead><tr><th>パラメータ</th><th>レベル</th><th>意味</th><th>将来調整</th></tr></thead>
      <tbody>
        <tr><td><code>alpha</code></td><td>品目</td><td>Adstock減衰率（訪問効果の持続性）</td><td>変えない（チャネル固有）</td></tr>
        <tr><td><code>beta_m</code></td><td>品目×チャネル</td><td>応答曲線の天井（最大売上効果）</td><td><strong>変える</strong>（LOE/発売で変化）</td></tr>
        <tr><td><code>ec_m</code></td><td>品目×チャネル</td><td>EC50（半最大効果に必要な活動量）</td><td>競合参入で低下</td></tr>
        <tr><td><code>eta_m</code></td><td>品目×チャネル</td><td>チャネル効率パラメータ</td><td>デジタル年率+2%</td></tr>
        <tr><td><code>slope_m</code></td><td>全チャネル共通</td><td>Hill係数 = <strong>1.0 固定</strong>（ミカエリス-メンテン型）</td><td>変えない</td></tr>
        <tr><td><code>gamma_c</code></td><td>施設クラスター</td><td>クラスター固有の主効果</td><td>変えない（施設特性は安定）</td></tr>
        <tr><td><code>gamma_gc</code></td><td>品目×施設クラスター</td><td>品目とクラスターの交互作用効果</td><td>ライフサイクルで変化</td></tr>
        <tr><td><code>sigma</code></td><td>品目</td><td>観測誤差・ノイズ</td><td>変えない</td></tr>
      </tbody>
    </table>

    <h3 style="margin-top:24px;">イベント別調整ルール</h3>
    <table class="data-table">
      <thead><tr><th>イベント</th><th>対象パラメータ</th><th>調整内容</th><th>根拠</th></tr></thead>
      <tbody>
        <tr><td>LOE（特許切れ）0年後</td><td>beta_m</td><td>×0.80</td><td>ジェネリック参入直後の処方シフト</td></tr>
        <tr><td>LOE 1年後</td><td>beta_m</td><td>×0.55</td><td>類似品LOE実績に基づく</td></tr>
        <tr><td>LOE 2年後</td><td>beta_m</td><td>×0.35</td><td>同上</td></tr>
        <tr><td>LOE 3年以降</td><td>beta_m</td><td>×0.20</td><td>最大減衰（バイオロジクスは緩やか）</td></tr>
        <tr><td>LOE 全期間</td><td>ec_m</td><td>×0.90→0.50</td><td>市場縮小で飽和点が早まる</td></tr>
        <tr><td>新製品発売 0年目</td><td>beta_m</td><td>×0.40</td><td>認知度低い段階</td></tr>
        <tr><td>新製品発売 1年目</td><td>beta_m</td><td>×0.60</td><td>ランプアップ中</td></tr>
        <tr><td>新製品発売 2年目</td><td>beta_m</td><td>×0.80</td><td>市場定着期</td></tr>
        <tr><td>新製品発売 3年目以降</td><td>beta_m</td><td>×1.00</td><td>フル効果（成熟期）</td></tr>
        <tr><td>競合参入</td><td>ec_m</td><td>×0.85</td><td>市場競争激化で早期飽和</td></tr>
        <tr><td>デジタルチャネル（毎年）</td><td>eta_m</td><td>×(1+0.02)^年数</td><td>デジタル普及率年率2%向上トレンド</td></tr>
      </tbody>
    </table>

    <h3 style="margin-top:24px;">レスポンスカーブ（slope_m=1）</h3>
    <div class="formula-box">
      <div class="formula-label">ミカエリス-メンテン型（slope=1）</div>
<pre>効果(x) = beta_m × x / (ec_m + x)

限界効果（微分）= beta_m × ec_m / (ec_m + x)²
  → LOE後: beta_m 低下 → 天井が下がる
  → 競合参入: ec_m 低下 → 少ない活動で飽和するが天井も低め</pre>
    </div>

    <h3 style="margin-top:24px;">Databricks 連携設計</h3>
    <table class="data-table">
      <thead><tr><th>フェーズ</th><th>方式</th><th>精度</th><th>工数</th></tr></thead>
      <tbody>
        <tr><td>現在</td><td>ダミーパラメータ（mmm_decay_params.csv）</td><td>低</td><td>—</td></tr>
        <tr><td>Phase 1</td><td>DatabricksからMAP推定値をCSV出力 → mmm_params_export.csv として読込</td><td><strong>高</strong></td><td>小</td></tr>
        <tr><td>Phase 2</td><td>Databricks REST API ライブ呼び出し（リアルタイム更新）</td><td>最高</td><td>大</td></tr>
      </tbody>
    </table>
    <p style="font-size:12px;color:#666;margin-top:8px;">
      Phase 1 CSV スキーマ例: <code>product_id, facility_cluster, channel, alpha, beta_m, ec_m, eta_m, mr_time_weight, slope_m</code><br>
      gamma_gc（品目×施設クラスター）を取り込むことで施設レベルのFTE最適配分が可能になる。
    </p>

  </div>
</div><!-- /mmm-lifecycle -->

<!-- ── DATA FILES ─────────────────────────────────────────── -->
<div class="section-card" id="data-files">
  <div class="section-header">
    <span class="step-badge" style="background:linear-gradient(135deg,#374151,#1f2937);">D</span>
    <div>
      <div class="section-title">データファイル一覧</div>
      <div class="section-subtitle">システムが参照する入力ファイルと出力ファイル</div>
    </div>
  </div>
  <div class="section-body">
    <table class="data-table">
      <thead>
        <tr>
          <th>#</th>
          <th>ファイル名</th>
          <th>種別</th>
          <th>内容・用途</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>1</td><td><code>products.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>品目マスタ（product_id, area, is_new, launch_date等）</td></tr>
        <tr><td>2</td><td><code>mindscape.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>品目×セグメント×期の医師数・ウェイト</td></tr>
        <tr><td>3</td><td><code>mmm_decay_params.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>MMMパラメータ（beta_m, ec_m, eta_m, slope_m=1固定）品目×7チャネル別</td></tr>
        <tr><td>4</td><td><code>digital_soc.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>デジタルSOC率の年次データ</td></tr>
        <tr><td>5</td><td><code>fc_rate.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>品目別FC率（月次）</td></tr>
        <tr><td>6</td><td><code>competition.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>競合環境係数（品目×月）</td></tr>
        <tr><td>7</td><td><code>hc_targets.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>CS/PSエリア別HCターゲット値（年度別）</td></tr>
        <tr><td>8</td><td><code>launch_requirements.csv</code></td><td><span class="tag tag-existing">入力</span></td><td>新発売品の発売月・必要FTE要件</td></tr>
        <tr><td>9</td><td><code>fte_output.csv</code></td><td><span class="tag tag-new">出力</span></td><td>品目×月別FTE（正規化後）</td></tr>
        <tr><td>10</td><td><code>fte_summary.csv</code></td><td><span class="tag tag-new">出力</span></td><td>エリア×年度集計サマリー</td></tr>
        <tr><td>11</td><td><code>digital_score.csv</code></td><td><span class="tag tag-new">出力</span></td><td>品目×月別デジタル有効性スコア</td></tr>
        <tr><td>12</td><td><code>fte_logic_document.html</code></td><td><span class="tag tag-new">出力</span></td><td>本ロジックドキュメント（HTML）</td></tr>
      </tbody>
    </table>
  </div>
</div><!-- /data-files -->

<div class="doc-footer">
  Generated by MR FTE Simulation System &nbsp;|&nbsp; GENERATED_AT_PLACEHOLDER
</div>

</div><!-- /content-wrap -->
</main>

<script>
// Smooth active state on sidebar links
(function() {
  var links = document.querySelectorAll('nav#sidebar a');
  var sections = [];
  links.forEach(function(link) {
    var href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      var el = document.getElementById(href.slice(1));
      if (el) sections.push({ el: el, link: link });
    }
  });
  function onScroll() {
    var scrollY = window.pageYOffset || document.documentElement.scrollTop;
    var main = document.querySelector('main');
    if (main) scrollY = main.scrollTop;
    var current = null;
    sections.forEach(function(s) {
      if (s.el.offsetTop - 80 <= scrollY) current = s;
    });
    links.forEach(function(l) { l.classList.remove('active'); });
    if (current) current.link.classList.add('active');
  }
  var main = document.querySelector('main');
  if (main) main.addEventListener('scroll', onScroll, { passive: true });
  else window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  // Smooth scroll
  links.forEach(function(link) {
    link.addEventListener('click', function(e) {
      var href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        e.preventDefault();
        var target = document.getElementById(href.slice(1));
        if (target) {
          var main = document.querySelector('main');
          if (main) {
            main.scrollTo({ top: target.offsetTop - 16, behavior: 'smooth' });
          } else {
            target.scrollIntoView({ behavior: 'smooth' });
          }
        }
      }
    });
  });
})();
</script>
</body>
</html>"""

    html = html.replace("GENERATED_AT_PLACEHOLDER", generated_at)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fte_logic_document.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] ロジックドキュメント出力: {out_path.resolve()}")
    return out_path


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

    # ----------------------------------------------------------
    # MMM レスポンスカーブ可視化
    # ----------------------------------------------------------

    def fig_mmm_response_curves(
        self,
        adjuster,  # MMMParameterAdjuster instance
        product_id: str,
        channels: Optional[List[str]] = None,
        fiscal_years: Optional[List[int]] = None,
    ) -> go.Figure:
        """
        品目×チャネルのMMMレスポンスカーブ（FY別）を描画する。

        Parameters
        ----------
        adjuster      : MMMParameterAdjuster インスタンス
        product_id    : 対象品目ID
        channels      : 対象チャネルリスト（デフォルト: 面談/面談_アポ/説明会）
        fiscal_years  : 対象年度リスト（デフォルト: [2026, 2029, 2032, 2035]）

        Returns
        -------
        go.Figure: Plotly Figure
        """
        if channels is None:
            channels = ["面談", "面談_アポ", "説明会"]
        if fiscal_years is None:
            fiscal_years = [2026, 2029, 2032, 2035]

        fy_colors = {2026: "#1f77b4", 2029: "#ff7f0e", 2032: "#2ca02c", 2035: "#d62728"}
        fy_dash   = {2026: "solid", 2029: "dash", 2032: "dot", 2035: "dashdot"}

        n_ch = len(channels)
        fig = make_subplots(
            rows=1,
            cols=n_ch,
            subplot_titles=channels,
            shared_yaxes=False,
        )

        for col_idx, ch in enumerate(channels, start=1):
            # 基準年度(FY2026)のec_mからx軸範囲を固定 → 全年度で同じ横軸
            base_pts = adjuster.response_curve_points(product_id, fiscal_years[0], ch)
            x_max_fixed = max((p[0] for p in base_pts), default=None)

            for fy in fiscal_years:
                pts = adjuster.response_curve_points(product_id, fy, ch, x_max=x_max_fixed)
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                show_legend = (col_idx == 1)
                fig.add_trace(
                    go.Scatter(
                        name=f"FY{fy}",
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(
                            color=fy_colors.get(fy, "#aaa"),
                            dash=fy_dash.get(fy, "solid"),
                            width=2,
                        ),
                        legendgroup=f"FY{fy}",
                        showlegend=show_legend,
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            title=f"{product_id} — チャネル別 MMMレスポンスカーブ（FY別ライフサイクル調整後）",
            height=380,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(title_text="活動量")
        fig.update_yaxes(title_text="売上効果")
        return fig

    def generate_mmm_section(
        self,
        decay_params_df: "pd.DataFrame",
        loe_schedule: Optional[Dict[str, str]] = None,
        launch_schedule: Optional[Dict[str, str]] = None,
        competitor_entry: Optional[Dict[str, str]] = None,
        products: Optional[List[str]] = None,
        fiscal_years: Optional[List[int]] = None,
    ) -> str:
        """
        MMMパラメータ調整セクションのHTML文字列を生成する。

        a) 品目別 MR チャネルレスポンスカーブ（FY2026/2029/2032/2035）
        b) パラメータ調整テーブル（品目×年度×チャネル）
        c) Databricks連携ノート

        Returns
        -------
        str: HTML文字列（<section>タグ含む）
        """
        from fy2029_fte_calculator import MMMParameterAdjuster

        if fiscal_years is None:
            fiscal_years = [2026, 2029, 2032, 2035]
        if products is None:
            products = list(decay_params_df["product_id"].unique()) if not decay_params_df.empty else []

        adjuster = MMMParameterAdjuster(
            decay_params_df=decay_params_df,
            loe_schedule=loe_schedule or {},
            launch_schedule=launch_schedule or {},
            competitor_entry=competitor_entry or {},
            base_year=2026,
        )

        mr_channels = ["面談", "面談_アポ", "説明会"]

        # ---- a) レスポンスカーブ ----
        curve_charts_html = ""
        for pid in products[:10]:  # 最大10品目表示
            try:
                fig = self.fig_mmm_response_curves(adjuster, pid, mr_channels, fiscal_years)
                curve_charts_html += f'<div style="margin-bottom:20px;">{_fig_to_html(fig)}</div>\n'
            except Exception as exc:
                curve_charts_html += f'<p style="color:#999;">{pid}: グラフ生成エラー ({exc})</p>\n'

        # ---- b) パラメータ調整テーブル ----
        adj_rows = []
        for pid in products:
            for ch in mr_channels:
                base_row = decay_params_df[
                    (decay_params_df["product_id"] == pid) & (decay_params_df["channel"] == ch)
                ]
                if base_row.empty:
                    continue
                base_beta = float(base_row["beta_m"].iloc[0])
                base_ec   = float(base_row["ec_m"].iloc[0])
                row_data: Dict[str, object] = {
                    "品目": pid,
                    "チャネル": ch,
                    "base beta_m": round(base_beta, 1),
                    "base ec_m": round(base_ec, 1),
                }
                for fy in fiscal_years:
                    adj = adjuster.get_adjusted_params(pid, fy)
                    ch_row = adj[adj["channel"] == ch]
                    if not ch_row.empty:
                        row_data[f"beta_m FY{fy}"] = round(float(ch_row["beta_m"].iloc[0]), 1)
                        row_data[f"ec_m FY{fy}"]   = round(float(ch_row["ec_m"].iloc[0]),   1)
                    else:
                        row_data[f"beta_m FY{fy}"] = "-"
                        row_data[f"ec_m FY{fy}"]   = "-"

                # 調整理由
                reasons = []
                if loe_schedule and pid in loe_schedule:
                    reasons.append(f"LOE({loe_schedule[pid]})")
                if launch_schedule and pid in launch_schedule:
                    reasons.append(f"新製品ランプアップ({launch_schedule[pid]})")
                if competitor_entry and pid in competitor_entry:
                    reasons.append(f"競合参入({competitor_entry[pid]})")
                row_data["調整理由"] = " / ".join(reasons) if reasons else "—"
                adj_rows.append(row_data)

        if adj_rows:
            adj_df = pd.DataFrame(adj_rows)
            param_table_html = _df_to_html(adj_df, title="MMMパラメータ ライフサイクル調整テーブル（MRチャネル）")
        else:
            param_table_html = "<p style='color:#999;'>パラメータデータなし</p>"

        # ---- c) Databricks連携ノート ----
        databricks_note = """
<div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:16px 20px;border-radius:0 8px 8px 0;margin-top:24px;">
  <div style="font-weight:700;font-size:14px;color:#1d4ed8;margin-bottom:8px;">Databricks 連携設計メモ</div>
  <ul style="font-size:13px;color:#374151;line-height:1.8;margin:0;padding-left:18px;">
    <li><strong>現在</strong>: mmm_decay_params.csv のダミーパラメータを使用</li>
    <li><strong>Phase 1</strong>: Databricks Meridian ジョブの事後分布 MAP 推定値を
      <code>mmm_params_export.csv</code> に出力 → 本システムが読み込み</li>
    <li><strong>Phase 2</strong>: Databricks REST API 経由でライブ取得（リアルタイム更新対応）</li>
    <li><strong>拡張効果</strong>: 品目×施設クラスタ (<code>gamma_gc</code>) パラメータを取り込むことで、
      施設レベルのFTE最適配分が可能になる（現行は品目レベル）</li>
  </ul>
  <div style="font-size:12px;color:#6b7280;margin-top:10px;">
    出力スキーマ例: product_id, facility_cluster, channel, alpha, beta_m, ec_m, eta_m, mr_time_weight, slope_m
  </div>
</div>
"""

        html = f"""
<section>
  <h2>⑪ MMMパラメータ ライフサイクル調整</h2>
  <p style="font-size:13px;color:#666;margin-bottom:12px;">
    MMMパラメータ（beta_m / ec_m / eta_m）をLOE・新製品発売・競合参入・デジタルトレンドに基づいて将来年度へ調整したレスポンスカーブを示す。<br>
    各曲線の傾きが品目への追加投資の限界効率を表し、傾きが均等になる点が最適FTE配分となる。
  </p>
  {curve_charts_html}
  {param_table_html}
  {databricks_note}
</section>
"""
        return html

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
        soc_rates: Optional[Dict[str, Dict[str, float]]] = None,
        digital_score_df: Optional[pd.DataFrame] = None,
        decay_params_df: Optional[pd.DataFrame] = None,
        loe_schedule: Optional[Dict[str, str]] = None,
        launch_schedule: Optional[Dict[str, str]] = None,
        competitor_entry: Optional[Dict[str, str]] = None,
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

        n_products = fte_df["product_id"].nunique()
        _score_df = digital_score_df if digital_score_df is not None else pd.DataFrame()
        avg_digital_score = _score_df["digital_score"].mean() if not _score_df.empty else 0.0

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
            _kpi_card("平均デジタル有効性", f"{avg_digital_score*100:.0f}%", "全品目スコア平均"),
        ])

        # ---- FC/SC サマリーテーブル（品目の訪問種別コスト重み） ----
        fc_sc_summary = (
            fte_df.groupby("product_id")
            .agg(
                avg_fc_ratio=("fc_ratio", "mean") if "fc_ratio" in fte_df.columns else ("target_doctors", "count"),
                avg_fc_weight=("fc_weight", "mean") if "fc_weight" in fte_df.columns else ("target_doctors", "count"),
                avg_total=("target_doctors", "mean"),
            )
            .reset_index()
            .round(3)
        )
        if "fc_ratio" in fte_df.columns:
            fc_sc_summary.columns = ["品目", "FC比率(平均)", "訪問コスト重み(平均)", "ターゲット医師数(平均)"]
        else:
            fc_sc_summary.columns = ["品目", "件数", "件数2", "ターゲット医師数(平均)"]

        # ---- 詳細テーブル用カラム整理 ----
        detail_cols = [
            "product_id", "month", "area",
            "target_doctors", "r_doctors", "w_doctors",
            "fc_ratio", "visit_frequency",
            "required_calls", fte_col,
        ]
        col_labels = [
            "品目", "月", "領域",
            "ターゲット医師数", "R医師数", "W医師数",
            "FC比率", "訪問頻度",
            "必要コール数", "必要FTE",
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

        # ---- SOC パラメータテーブル ----
        if soc_rates:
            _soc_rows = []
            for pid, rates in sorted(soc_rates.items()):
                _soc_rows.append({
                    "品目": pid,
                    "MR想起率": rates.get("mr", 0.50),
                    "デジタル想起率": rates.get("digital", 0.25),
                })
            soc_df = pd.DataFrame(_soc_rows)
            # SOC活動数から有効接触も追加（summary_latestから引用）
            soc_table_html = _df_to_html(
                soc_df,
                title="品目別 SOCパラメータ（想起率）",
                highlight_cols=["MR想起率", "デジタル想起率"],
            )
        else:
            soc_table_html = "<p style='color:#999;'>SOCデータなし</p>"

        # ---- MMM レスポンスカーブセクション ----
        if decay_params_df is not None and not decay_params_df.empty:
            mmm_html = self.generate_mmm_section(
                decay_params_df,
                loe_schedule=loe_schedule,
                launch_schedule=launch_schedule,
                competitor_entry=competitor_entry,
            )
        else:
            mmm_html = "<p style='color:#999;padding:20px;'>MMMパラメータCSV未設定</p>"

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
            chart_digital_score=_fig_to_html(
                fig_digital_effectiveness(digital_score_df if digital_score_df is not None else pd.DataFrame()),
                "chart_digital_score",
            ),
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
                bar_cols=["avg_required_fte", "max_required_fte"],
            ),
            table_fc_sc=_df_to_html(fc_sc_summary, title="FC/SC訪問コスト重み内訳"),
            table_ove=_df_to_html(allocation_df, title="ドナー品目別 削減FTE詳細",
                                   bar_cols=["fte_reduction"]),
            table_headcount=_df_to_html(total_fte_display, title="領域×月別 FTE過不足一覧"),
            chart_digital_activity=_fig_to_html(fig_digital_activity(_digital_df), "chart_digital_activity"),
            chart_digital_trend=_fig_to_html(fig_digital_trend(_digital_df), "chart_digital_trend"),
            table_digital_summary=digital_summary_html,
            mmm_section=mmm_html,
            table_soc_params=soc_table_html,
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
