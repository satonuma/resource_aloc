"""
FY2029 FTE算出 感度分析モジュール（続き①）
==========================================
売上予測の変化・活動頻度の仮定に対するFTEの感度を分析する。

シナリオ:
  BASE    : 基準ケース（lifecycle_adjusted頻度）
  HIGH    : 売上予測 +20%（ターゲット医師数 +20%）
  LOW     : 売上予測 -20%（ターゲット医師数 -20%）
  MMM_OPT : MMM最適頻度ベース（裏計算・参考値）
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fy2029_fte_calculator import (
    FY2029FTECalculator,
    FY2029_MONTHS,
    CURRENT_MR_COUNT,
    ProductConfig,
    TargetDoctorCalculator,
    FCScAllocator,
    ActivityFrequencyEstimator,
    DigitalEffectivenessScorer,
    NewProductFTEAllocator,
    normalize_fte_to_headcount,
)


# ============================================================
# シナリオ定義
# ============================================================

SCENARIOS = {
    "BASE":    {"label": "基準ケース",           "doctor_scale": 1.00, "color": "#1f77b4"},
    "HIGH":    {"label": "売上予測 +20%",        "doctor_scale": 1.20, "color": "#2ca02c"},
    "LOW":     {"label": "売上予測 -20%",        "doctor_scale": 0.80, "color": "#d62728"},
    "MMM_OPT": {"label": "MMM最適頻度（参考）", "doctor_scale": 1.00, "color": "#9467bd"},
}


# ============================================================
# ターゲット医師数スケーリング用 ラッパー
# ============================================================

class ScaledTargetDoctorCalculator(TargetDoctorCalculator):
    """ターゲット医師数に一律スケール係数を掛けるラッパー"""

    def __init__(self, base_calc: TargetDoctorCalculator, scale: float = 1.0) -> None:
        # 親クラスの初期化をスキップしてbaseを保持
        self._base = base_calc
        self.scale = scale
        # 親クラス属性を委譲
        self.activity_data         = base_calc.activity_data
        self.product_info          = base_calc.product_info
        self.base_target_doctors   = base_calc.base_target_doctors
        self.mindscape_segments_df = base_calc.mindscape_segments_df
        self.doctor_tiers          = base_calc.doctor_tiers

    def calculate(self, product_id, months, config=None, ramp_up_curve=None,
                  reference_product_id=None):
        df = self._base.calculate(
            product_id, months, config=config,
            ramp_up_curve=ramp_up_curve,
            reference_product_id=reference_product_id,
        )
        df = df.copy()
        df["target_doctors"] = (df["target_doctors"] * self.scale).round(0).astype(int)
        return df

    def get_base_target_doctors(self, product_id):
        base = self._base.get_base_target_doctors(product_id)
        return int(base * self.scale)

    def get_doctor_tier(self, product_id):
        tier = self._base.get_doctor_tier(product_id)
        if tier is None:
            return None
        return {
            "r_doctors":    int(round(tier["r_doctors"] * self.scale)),
            "w_doctors":    int(round(tier["w_doctors"] * self.scale)),
            "r_visit_freq": tier["r_visit_freq"],
            "w_visit_freq": tier["w_visit_freq"],
        }


# ============================================================
# 感度分析クラス
# ============================================================

class FY2029SensitivityAnalyzer:
    """
    売上予測の楽観/悲観シナリオ × 活動頻度の仮定に対するFTE感度分析。

    使い方:
        analyzer = FY2029SensitivityAnalyzer(
            base_calculator=calculator,
            base_target_calc=target_doctor_calc,
            ...
        )
        results = analyzer.run_all_scenarios()
        fig = analyzer.fig_scenario_comparison(results)
    """

    def __init__(
        self,
        base_calculator: FY2029FTECalculator,
        base_target_calc: TargetDoctorCalculator,
        mmm_freq_estimator: Optional[ActivityFrequencyEstimator] = None,
    ) -> None:
        self.base_calc = base_calculator
        self.base_target_calc = base_target_calc
        self.mmm_freq_estimator = mmm_freq_estimator

    def run_scenario(
        self,
        scenario_key: str,
        new_product_ids: Optional[List[str]] = None,
        donor_products: Optional[List[str]] = None,
        new_product_fte_allocator: Optional[NewProductFTEAllocator] = None,
    ) -> pd.DataFrame:
        """
        指定シナリオのFTEを算出して返す。
        カラム: product_id, month, required_fte, scenario
        """
        sc = SCENARIOS[scenario_key]
        scale = sc["doctor_scale"]

        # ターゲット医師数をスケーリング
        scaled_target_calc = ScaledTargetDoctorCalculator(self.base_target_calc, scale)

        # 頻度推定器
        if scenario_key == "MMM_OPT" and self.mmm_freq_estimator is not None:
            freq_est = self.mmm_freq_estimator
            freq_mode = "mmm_optimal"
        else:
            freq_est = self.base_calc.freq_estimator
            freq_mode = self.base_calc.frequency_mode

        # 新しい Calculator インスタンスを構築
        calc = FY2029FTECalculator(
            product_configs=list(self.base_calc.configs.values()),
            target_doctor_calc=scaled_target_calc,
            fc_sc_allocator=self.base_calc.fc_sc_allocator,
            freq_estimator=freq_est,
            product_info=self.base_calc.product_info,
            current_activities=self.base_calc.current_activities,
            frequency_mode=freq_mode,
            new_product_ramp_up=self.base_calc.new_product_ramp_up,
            reference_products=self.base_calc.reference_products,
            target_months=self.base_calc._target_months,
        )

        if new_product_ids and donor_products and new_product_fte_allocator:
            fte_df, _ = calc.run_with_new_product(
                new_product_ids, donor_products, new_product_fte_allocator
            )
        else:
            fte_df = calc.run()

        # 基準ケースと同じヘッドカウント制約に正規化（CS=380, PS=45）
        fte_df = normalize_fte_to_headcount(fte_df, CURRENT_MR_COUNT)

        fte_col = "adjusted_fte" if "adjusted_fte" in fte_df.columns else "required_fte"
        fte_df["scenario"] = scenario_key
        fte_df["scenario_label"] = sc["label"]
        fte_df["fte"] = fte_df[fte_col]
        return fte_df

    def run_all_scenarios(
        self,
        new_product_ids: Optional[List[str]] = None,
        donor_products: Optional[List[str]] = None,
        new_product_fte_allocator: Optional[NewProductFTEAllocator] = None,
    ) -> Dict[str, pd.DataFrame]:
        """全シナリオを実行して辞書で返す"""
        results = {}
        for key in SCENARIOS:
            if key == "MMM_OPT" and self.mmm_freq_estimator is None:
                continue
            print(f"  シナリオ実行: {key} ({SCENARIOS[key]['label']})")
            results[key] = self.run_scenario(
                key, new_product_ids, donor_products, new_product_fte_allocator
            )
        return results

    # ----------------------------------------------------------
    # 集計・出力
    # ----------------------------------------------------------

    def summarize_scenarios(
        self, results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        シナリオ×品目×年度のFTEサマリー（横持ち）

        Returns
        -------
        DataFrame: product_id, area, BASE_avg_fte, HIGH_avg_fte, LOW_avg_fte, ...
        """
        rows = []
        for key, df in results.items():
            fte_col = "fte"
            agg = (
                df.groupby(["product_id", "area"])[fte_col]
                .agg(avg_fte="mean", max_fte="max")
                .reset_index()
            )
            agg["scenario"] = key
            rows.append(agg)

        all_df = pd.concat(rows, ignore_index=True)

        # 横持ちに変換
        pivot = all_df.pivot_table(
            index=["product_id", "area"],
            columns="scenario",
            values="avg_fte",
        ).reset_index()
        pivot.columns.name = None

        # シナリオ列名を整理
        for key in SCENARIOS:
            if key in pivot.columns:
                pivot.rename(columns={key: f"{key}_avg_fte"}, inplace=True)

        # BASE比の変化率を追加
        if "BASE_avg_fte" in pivot.columns:
            for key in ["HIGH", "LOW", "MMM_OPT"]:
                col = f"{key}_avg_fte"
                if col in pivot.columns:
                    pivot[f"{key}_vs_BASE"] = (
                        (pivot[col] - pivot["BASE_avg_fte"])
                        / pivot["BASE_avg_fte"].replace(0, float("nan"))
                    ).round(3)

        return pivot.round(2)

    def total_by_area_scenario(
        self, results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """領域×月×シナリオの合計FTE"""
        rows = []
        for key, df in results.items():
            agg = (
                df.groupby(["area", "month"])["fte"]
                .sum()
                .reset_index()
                .rename(columns={"fte": "total_fte"})
            )
            agg["scenario"] = key
            agg["scenario_label"] = SCENARIOS[key]["label"]
            rows.append(agg)
        return pd.concat(rows, ignore_index=True)

    # ----------------------------------------------------------
    # グラフ生成
    # ----------------------------------------------------------

    def fig_scenario_comparison(
        self, results: Dict[str, pd.DataFrame], area: str = "CS"
    ) -> go.Figure:
        """
        領域ごとの合計FTE月次推移（シナリオ別）+ 現行MR水平線
        """
        total_df = self.total_by_area_scenario(results)
        sub = total_df[total_df["area"] == area].sort_values(["scenario", "month"])

        fig = go.Figure()

        for key, sc in SCENARIOS.items():
            df_sc = sub[sub["scenario"] == key]
            if df_sc.empty:
                continue
            fig.add_trace(go.Scatter(
                name=sc["label"],
                x=df_sc["month"],
                y=df_sc["total_fte"],
                mode="lines+markers",
                line=dict(color=sc["color"], width=2,
                          dash="dot" if key == "MMM_OPT" else "solid"),
                marker=dict(size=5),
            ))

        # 現行MR数（水平線）
        fig.add_hline(
            y=CURRENT_MR_COUNT.get(area, 0),
            line_dash="dash",
            line_color="black",
            annotation_text=f"現行{area}MR数 ({CURRENT_MR_COUNT.get(area, 0)}名)",
            annotation_position="right",
        )

        fig.update_layout(
            title=f"感度分析: {area}領域 合計FTE シナリオ比較 - FY2029",
            xaxis_title="月",
            yaxis_title="合計FTE",
            height=420,
            template="plotly_white",
            legend=dict(orientation="h", y=1.08),
        )
        fig.update_xaxes(tickangle=45)
        return fig

    def fig_product_scenario_heatmap(
        self, results: Dict[str, pd.DataFrame]
    ) -> go.Figure:
        """
        品目×シナリオの平均FTE ヒートマップ
        """
        summary = self.summarize_scenarios(results)
        scenario_cols = [f"{k}_avg_fte" for k in SCENARIOS if f"{k}_avg_fte" in summary.columns]

        z = summary[scenario_cols].values
        x_labels = [c.replace("_avg_fte", "") for c in scenario_cols]
        y_labels = summary["product_id"].tolist()

        fig = go.Figure(go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            text=z.round(1),
            texttemplate="%{text}",
            hovertemplate="品目: %{y}<br>シナリオ: %{x}<br>平均FTE: %{z:.1f}<extra></extra>",
        ))
        fig.update_layout(
            title="品目×シナリオ 平均FTE ヒートマップ - FY2029",
            xaxis_title="シナリオ",
            yaxis_title="品目",
            height=350,
            template="plotly_white",
        )
        return fig

    def fig_gap_waterfall(
        self, results: Dict[str, pd.DataFrame], area: str = "CS"
    ) -> go.Figure:
        """
        BASE vs HIGH/LOW シナリオのFTE差異をウォーターフォールで表示
        """
        summary = self.summarize_scenarios(results)
        sub = summary[summary["area"] == area].copy()

        products = sub["product_id"].tolist()
        base_vals = sub.get("BASE_avg_fte", pd.Series([0]*len(sub))).tolist()
        high_vals = sub.get("HIGH_avg_fte", base_vals)
        low_vals  = sub.get("LOW_avg_fte",  base_vals)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="基準ケース",
            x=products,
            y=base_vals,
            marker_color="#1f77b4",
        ))
        fig.add_trace(go.Bar(
            name="売上+20%",
            x=products,
            y=[h - b for h, b in zip(high_vals, base_vals)],
            base=base_vals,
            marker_color="#2ca02c",
            opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            name="売上-20%",
            x=products,
            y=[l - b for l, b in zip(low_vals, base_vals)],
            base=base_vals,
            marker_color="#d62728",
            opacity=0.7,
        ))

        fig.update_layout(
            barmode="overlay",
            title=f"感度分析: {area}領域 品目別FTEレンジ（HIGH/BASE/LOW）",
            xaxis_title="品目",
            yaxis_title="平均FTE",
            height=380,
            template="plotly_white",
        )
        return fig

    def export_sensitivity_html(
        self,
        results: Dict[str, pd.DataFrame],
        output_dir: str = "output",
        filename: str = "fy2029_sensitivity.html",
    ) -> str:
        """感度分析専用HTMLを出力"""
        from pathlib import Path
        import plotly.io as pio

        out = Path(output_dir) / filename
        summary_df = self.summarize_scenarios(results)

        figs_html = ""
        for area in ["CS", "PS"]:
            figs_html += f"<h2>{area}領域 シナリオ比較</h2>"
            figs_html += self.fig_scenario_comparison(results, area).to_html(
                full_html=False, include_plotlyjs=False
            )
        figs_html += "<h2>品目×シナリオ ヒートマップ</h2>"
        figs_html += self.fig_product_scenario_heatmap(results).to_html(
            full_html=False, include_plotlyjs=False
        )
        figs_html += "<h2>CS領域 FTEレンジ</h2>"
        figs_html += self.fig_gap_waterfall(results, "CS").to_html(
            full_html=False, include_plotlyjs=False
        )

        table_html = summary_df.style.format(precision=2).to_html()

        html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>FY2029 感度分析レポート</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: "Segoe UI", sans-serif; max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ background: #2c3e50; color: white; padding: 16px 24px; border-radius: 8px; }}
    h2 {{ margin: 24px 0 12px; border-left: 4px solid #3498db; padding-left: 10px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{ background: #2c3e50; color: white; padding: 8px 12px; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #dee2e6; }}
    tr:hover td {{ background: #f0f4ff; }}
  </style>
</head>
<body>
  <h1>FY2029 感度分析レポート</h1>
  {figs_html}
  <h2>シナリオ×品目 FTEサマリー（横持ち）</h2>
  {table_html}
</body>
</html>"""

        out.write_text(html, encoding="utf-8")
        print(f"[OK] 感度分析レポート出力: {out.resolve()}")
        return str(out)
