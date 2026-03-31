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
      <strong>①MMM デジタル応答比（W=50%）</strong>：Hill関数の総効果量 dig_val/(mr_val+dig_val)。活動数補正済みのデジタルチャネル効果量。<br>
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
<title>FTE算出ロジックドキュメント (FY2026-2035)</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Segoe UI", "Helvetica Neue", Arial, "Hiragino Sans", sans-serif;
    background: #f4f6f9;
    color: #1a1a2e;
    display: flex;
    min-height: 100vh;
  }

  /* Sidebar */
  nav#sidebar {
    width: 240px;
    min-width: 240px;
    background: #0d1b2a;
    color: #c9d6df;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    padding: 24px 0;
    flex-shrink: 0;
  }
  nav#sidebar h1 {
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    padding: 0 20px 16px 20px;
    border-bottom: 1px solid #1e3a5f;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  nav#sidebar ul {
    list-style: none;
    padding: 12px 0;
  }
  nav#sidebar ul li a {
    display: block;
    padding: 9px 20px;
    color: #94a3b8;
    text-decoration: none;
    font-size: 13px;
    transition: background 0.15s, color 0.15s;
    border-left: 3px solid transparent;
  }
  nav#sidebar ul li a:hover {
    background: #1e3a5f;
    color: #e2e8f0;
    border-left-color: #3b82f6;
  }
  .nav-section {
    display: block;
    padding: 14px 20px 4px 20px;
    font-size: 10px;
    font-weight: 700;
    color: #4a6fa5;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  /* Main content */
  main {
    flex: 1;
    padding: 40px 48px;
    max-width: 960px;
  }

  .page-title {
    font-size: 26px;
    font-weight: 700;
    color: #0d1b2a;
    margin-bottom: 6px;
  }
  .page-subtitle {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 36px;
  }

  section {
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    padding: 28px 32px;
    margin-bottom: 28px;
  }
  section h2 {
    font-size: 18px;
    font-weight: 700;
    color: #0d1b2a;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 2px solid #3b82f6;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .step-badge {
    background: #3b82f6;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 12px;
    white-space: nowrap;
  }

  h3 {
    font-size: 14px;
    font-weight: 700;
    color: #1e3a5f;
    margin: 18px 0 8px 0;
  }
  p {
    font-size: 14px;
    line-height: 1.7;
    color: #374151;
    margin-bottom: 10px;
  }
  ul.logic-list {
    list-style: none;
    padding: 0;
    margin-bottom: 12px;
  }
  ul.logic-list li {
    font-size: 14px;
    line-height: 1.7;
    color: #374151;
    padding: 4px 0 4px 18px;
    position: relative;
  }
  ul.logic-list li::before {
    content: "\25B8";
    position: absolute;
    left: 0;
    color: #3b82f6;
    font-size: 12px;
  }

  .formula-box {
    background: #f0f4ff;
    border: 1px solid #bfcfff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 14px 0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 13px;
    color: #1e3a5f;
    line-height: 1.8;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .info-box {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    border-left: 4px solid #10b981;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 13px;
    color: #064e3b;
    line-height: 1.6;
  }

  table.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 10px;
  }
  table.data-table thead th {
    background: #0d1b2a;
    color: #e2e8f0;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }
  table.data-table tbody tr:nth-child(odd) {
    background: #f8fafc;
  }
  table.data-table tbody tr:nth-child(even) {
    background: #ffffff;
  }
  table.data-table tbody td {
    padding: 9px 14px;
    border-bottom: 1px solid #e2e8f0;
    vertical-align: top;
  }
  table.data-table tbody td:first-child {
    font-family: "Consolas", "Courier New", monospace;
    color: #1e3a5f;
    font-weight: 600;
    white-space: nowrap;
  }

  footer {
    margin-top: 40px;
    padding: 16px 0;
    font-size: 12px;
    color: #94a3b8;
    border-top: 1px solid #e2e8f0;
  }
</style>
</head>
<body>

<nav id="sidebar">
  <h1>FTE ロジック</h1>
  <ul>
    <li><span class="nav-section">ステップ</span></li>
    <li><a href="#step1">STEP 1: ターゲット医師数</a></li>
    <li><a href="#step2">STEP 2: FC/SC分割</a></li>
    <li><a href="#step3">STEP 3: 訪問頻度推定</a></li>
    <li><a href="#step4">STEP 4: MR FTE算出</a></li>
    <li><a href="#step5">STEP 5: HC正規化</a></li>
    <li><a href="#step6">STEP 6: 新発売品FTE配分</a></li>
    <li><span class="nav-section">別途分析</span></li>
    <li><a href="#digital">デジタル有効性スコア</a></li>
    <li><span class="nav-section">参照</span></li>
    <li><a href="#datafiles">データファイル一覧</a></li>
  </ul>
</nav>

<main>
  <div class="page-title">FTE算出ロジックドキュメント</div>
  <div class="page-subtitle">FY2026&#8211;2035 MR FTEシミュレーション &#47; 現行システム設計仕様</div>

  <!-- STEP 1 -->
  <section id="step1">
    <h2><span class="step-badge">STEP 1</span>ターゲット医師数</h2>
    <h3>既存品目</h3>
    <p><code>target_doctors.csv</code> から直読みする（過去活動データからの集計は廃止）。</p>

    <h3>新製品</h3>
    <p><code>mindscape_segments.csv</code> と <code>products.csv</code> の <code>mindscape_target_pct</code> 列を組み合わせて算出する。</p>
    <ul class="logic-list">
      <li>セグメント定義: (pct_from, pct_to, doctor_count) の行で構成</li>
      <li>条件: pct_to &lt;= mindscape_target_pct のセグメントの doctor_count を合計</li>
    </ul>
    <div class="formula-box">例（top % &#8594; 医師数）:
  OVE  (top 40%) &#8594; 1,100名
  Zaso (top 20%) &#8594;   300名
  WSA  (top 60%) &#8594; 1,190名
  GLO  (top 40%) &#8594;   720名</div>

    <h3>フォールバック（mindscape_segments未定義時）</h3>
    <div class="formula-box">target_doctors = ref_product_doctors &#215; (患者数比) &#215; &#8730;(効能数比)</div>

    <h3>ランプアップカーブ</h3>
    <p>発売後月数に応じた浸透率を乗算し、市場導入期の段階的な医師到達を表現する。</p>
  </section>

  <!-- STEP 2 -->
  <section id="step2">
    <h2><span class="step-badge">STEP 2</span>FC / SC 分割</h2>
    <ul class="logic-list">
      <li><strong>FC（ファーストコール）</strong>: 主訪問品目</li>
      <li><strong>SC（セカンドコール）</strong>: FC訪問に付随。FTEコスト = FC &#215; SC_COEFFICIENT（= 0.1）</li>
    </ul>
    <h3>SC &#8594; FC 格上げルール</h3>
    <p>SC品目のうち、FC品目の医師リストと重複しない医師はSCとして活動できないため、FCに格上げする。</p>
    <div class="formula-box">品目間被り率 = |FCリスト &#8745; SCリスト| / |SCリスト|</div>
    <h3>実効FC比率</h3>
    <ul class="logic-list">
      <li>優先: <code>fc_sc_ratio.csv</code> で明示指定された値を使用</li>
      <li>代替: 上記の被り率から自動計算</li>
    </ul>
  </section>

  <!-- STEP 3 -->
  <section id="step3">
    <h2><span class="step-badge">STEP 3</span>訪問頻度推定</h2>
    <ul class="logic-list">
      <li><strong>FC</strong>: ライフサイクル調整済み訪問頻度（発売・LOE・競合ブーストを反映）</li>
      <li><strong>SC</strong>: FC訪問に内包 &#8594; 追加コールなし（FTEコストのみ 0.1倍）</li>
    </ul>
    <p>競合品発売スケジュールは <code>competitor_schedule.csv</code> を参照してFTEブーストを算出する。</p>
  </section>

  <!-- STEP 4 -->
  <section id="step4">
    <h2><span class="step-badge">STEP 4</span>MR FTE算出（デジタル比率不使用）</h2>
    <div class="formula-box">FTE = (FC医師数 &#215; FC訪問頻度 + SC医師数 &#215; SC_COEFFICIENT)
    &#247; (稼働日[20日/月] &#215; コール/日)</div>
    <table class="data-table" style="margin-top:14px;width:auto;">
      <thead>
        <tr><th>チーム</th><th>コール/日</th><th>HC上限</th></tr>
      </thead>
      <tbody>
        <tr><td>CS（一般MR）</td><td>2.5</td><td>380名</td></tr>
        <tr><td>PS（専門MR）</td><td>1.5</td><td>45名</td></tr>
      </tbody>
    </table>
    <div class="info-box">デジタルはFTE計算に影響しない。MR 380名CS / 45名PSはMR活動のみで決まる。</div>
  </section>

  <!-- STEP 5 -->
  <section id="step5">
    <h2><span class="step-badge">STEP 5</span>ヘッドカウント正規化</h2>
    <ul class="logic-list">
      <li>月次・領域別FTE合計を CS=380、PS=45 にスケーリング</li>
      <li>品目間の相対比は保持される</li>
      <li>半期（H1: 4&#8211;9月、H2: 10&#8211;3月）ごとにステップ関数化</li>
    </ul>
    <div class="formula-box">normalized_FTE_i = raw_FTE_i &#215; (HC_target / &#931;raw_FTE_all)</div>
  </section>

  <!-- STEP 6 -->
  <section id="step6">
    <h2><span class="step-badge">STEP 6</span>新発売品FTE配分</h2>
    <ul class="logic-list">
      <li><code>fy2026_apr_fte</code> = FY2026/4時点の実績FTE（削減上限として使用）</li>
      <li>限界ROI = beta &#215; hill_marginal(x, ec, slope) が低い品目ほど優先的に削減</li>
      <li>削減下限: fy2026_apr_fte &#215; 0.5（最低50%保護）</li>
      <li>新品目の成長（ランチカーブ）に比例して月別段階的に移動</li>
    </ul>
    <div class="formula-box">限界ROI = beta &#215; hill_marginal(x, ec, slope)
削減量_上限 = current_fte &#8722; max(fy2026_apr_fte &#215; 0.5, target_fte)</div>
    <p>Hill関数パラメータ (beta, ec, slope) は <code>mmm_decay_params.csv</code> から参照する。</p>
  </section>

  <!-- Digital -->
  <section id="digital">
    <h2>別途分析: デジタル有効性スコア（FTEとは独立）</h2>
    <p>品目ごとのデジタル戦略的価値を 0&#8211;1 でスコア化する。FTE算出には影響しない。</p>

    <h3>スコア構成要素</h3>
    <table class="data-table" style="margin-top:10px;">
      <thead>
        <tr><th>指標</th><th>重み</th><th>内容</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>MMM デジタル応答比</td>
          <td>50%</td>
          <td>dig_hill_value / (mr_hill_value + dig_hill_value)</td>
        </tr>
        <tr>
          <td>SOC デジタル感受性</td>
          <td>50%</td>
          <td>digital_soc_rate（視聴1回あたりの想起確率）</td>
        </tr>
      </tbody>
    </table>

    <h3>ライフサイクル補正</h3>
    <table class="data-table" style="margin-top:10px;">
      <thead>
        <tr><th>条件</th><th>補正値</th></tr>
      </thead>
      <tbody>
        <tr><td>LOE後</td><td>&#8722;25 pt</td></tr>
        <tr><td>LOE まで &lt; 1年</td><td>&#8722;20 pt</td></tr>
        <tr><td>LOE まで 1&#8211;3年</td><td>&#8722;10 pt</td></tr>
        <tr><td>発売 &lt; 1年</td><td>&#8722;5 pt</td></tr>
        <tr><td>成長期</td><td>+8 pt</td></tr>
        <tr><td>成熟期</td><td>+5 pt</td></tr>
      </tbody>
    </table>

    <h3>スコア判定</h3>
    <table class="data-table" style="margin-top:10px;">
      <thead>
        <tr><th>スコア</th><th>判定</th><th>推奨アクション</th></tr>
      </thead>
      <tbody>
        <tr><td>&#8805; 55%</td><td>高</td><td>m3等デジタル積極活用</td></tr>
        <tr><td>35&#8211;55%</td><td>中</td><td>選択的デジタル活用</td></tr>
        <tr><td>&lt; 35%</td><td>低</td><td>MR中心維持</td></tr>
      </tbody>
    </table>
  </section>

  <!-- Data files -->
  <section id="datafiles">
    <h2>データファイル一覧</h2>
    <table class="data-table">
      <thead>
        <tr><th>ファイル名</th><th>用途</th></tr>
      </thead>
      <tbody>
        <tr><td>target_doctors.csv</td><td>既存品目のターゲット医師数（正値）</td></tr>
        <tr><td>mindscape_segments.csv</td><td>product_id, pct_from, pct_to, doctor_count（新製品用）</td></tr>
        <tr><td>products.csv</td><td>品目設定（mindscape_target_pct 列含む）</td></tr>
        <tr><td>fc_sc_ratio.csv</td><td>FC比率明示指定</td></tr>
        <tr><td>activity_set.csv</td><td>FC/SC品目セット</td></tr>
        <tr><td>target_doctor_ranges.csv</td><td>医師IDレンジ（被り率計算用）</td></tr>
        <tr><td>mmm_decay_params.csv</td><td>Hill関数パラメータ (beta, ec, slope)</td></tr>
        <tr><td>current_fte.csv</td><td>FY2026/4時点実績FTE（fy2026_apr_fte）</td></tr>
        <tr><td>soc_params.csv</td><td>品目別MR/デジタル想起率（SOCスコア用）</td></tr>
        <tr><td>current_activities.csv</td><td>月次MR/デジタル活動量</td></tr>
        <tr><td>competitor_schedule.csv</td><td>競合品発売スケジュール（FTEブースト）</td></tr>
        <tr><td>supply_restriction.csv</td><td>供給制限スケジュール</td></tr>
      </tbody>
    </table>
  </section>

  <footer>
    FTE算出システム &#47; 生成: GENERATED_AT_PLACEHOLDER
  </footer>
</main>

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
            table_fc_sc=_df_to_html(fc_sc_summary, title="FC/SC医師数内訳"),
            table_ove=_df_to_html(allocation_df, title="ドナー品目別 削減FTE詳細",
                                   bar_cols=["fte_reduction"]),
            table_headcount=_df_to_html(total_fte_display, title="領域×月別 FTE過不足一覧"),
            chart_digital_activity=_fig_to_html(fig_digital_activity(_digital_df), "chart_digital_activity"),
            chart_digital_trend=_fig_to_html(fig_digital_trend(_digital_df), "chart_digital_trend"),
            table_digital_summary=digital_summary_html,
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
