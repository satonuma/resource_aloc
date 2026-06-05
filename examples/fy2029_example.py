"""
FY2029 FTE算出 基本使用例
=========================
src/ のモジュールを使った最小限の動作確認用スクリプト。
Databricks環境では pd.read_csv → spark.read.table().toPandas() に置き換える。

実行方法（プロジェクトルートから）:
  python examples/fy2029_example.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# src/ をインポートパスに追加
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import numpy as np

from fy2029_fte_calculator import (
    ProductConfig,
    TargetDoctorCalculator,
    FCScAllocator,
    ActivityFrequencyEstimator,
    NewProductFTEAllocator,
    FY2029FTECalculator,
    get_fy_months,
    normalize_fte_to_headcount,
    CURRENT_MR_COUNT,
)

# ============================================================
# ダミーデータ定義
# ============================================================

# 品目設定（CS領域 4品目）
product_configs = [
    ProductConfig("GLI", "CS", False, "2018-04", 180, 50000, 3),
    ProductConfig("INT", "CS", False, "2019-07", 168, 80000, 4),
    ProductConfig("TRI", "CS", False, "2021-01", 120, 60000, 3),
    ProductConfig("OVE", "CS", True,  "2026-07", 240, 15000, 1),
]

# 品目情報 DataFrame（calculator が内部で参照）
product_info = pd.DataFrame([
    {
        "product_id":         cfg.product_id,
        "launch_ym":          cfg.launch_ym,
        "loe_months":         cfg.loe_months,
        "estimated_patients": cfg.estimated_patients,
        "num_indications":    cfg.num_indications,
        "post_loe_factor":    0.0,
    }
    for cfg in product_configs
])

# R/W ティア別 医師数・訪問頻度
doctor_tiers = {
    "GLI": {"r_doctors": 400, "w_doctors": 800, "r_visit_freq": 2.0, "w_visit_freq": 1.0},
    "INT": {"r_doctors": 500, "w_doctors": 1000, "r_visit_freq": 2.0, "w_visit_freq": 1.0},
    "TRI": {"r_doctors": 450, "w_doctors": 900,  "r_visit_freq": 2.0, "w_visit_freq": 1.0},
}
base_target_doctors = {
    pid: t["r_doctors"] + t["w_doctors"] for pid, t in doctor_tiers.items()
}

# 活動データ（2ヶ月分のダミー）
activity_data = pd.DataFrame({
    "activity_ym":    ["2024-04", "2024-04", "2024-05", "2024-05"] * 3,
    "product_id":     ["GLI"] * 4 + ["INT"] * 4 + ["TRI"] * 4,
    "doctor_id":      ["D001", "D002", "D001", "D003"] * 3,
    "activity_count": [2, 1, 2, 1] * 3,
})

# FC/SC 比率（1.0 = 全訪問がFC）
fc_ratios = {"GLI": 1.0, "INT": 0.8, "TRI": 0.6, "OVE": 1.0}

# OVE ランチカーブ（12ヶ月）: 発売直後が最大投資、徐々に逓減
ove_ramp_up = [1.00, 0.90, 0.82, 0.75, 0.70, 0.66, 0.63, 0.61, 0.59, 0.57, 0.56, 0.55]

# 計算対象月（FY2026 = 2026-04〜2027-03）
target_months = get_fy_months(2026)

# ============================================================
# モジュール初期化
# ============================================================

target_doctor_calc = TargetDoctorCalculator(
    base_target_doctors=base_target_doctors,
    product_info=product_info,
    activity_data=activity_data,
    doctor_tiers=doctor_tiers,
)

fc_sc_allocator = FCScAllocator(fc_ratios=fc_ratios)

freq_estimator = ActivityFrequencyEstimator(
    activity_data=activity_data,
    product_info=product_info,
)

# ============================================================
# FTE 算出
# ============================================================

calculator = FY2029FTECalculator(
    product_configs=product_configs,
    target_doctor_calc=target_doctor_calc,
    fc_sc_allocator=fc_sc_allocator,
    freq_estimator=freq_estimator,
    product_info=product_info,
    frequency_mode="lifecycle_adjusted",
    new_product_ramp_up={"OVE": ove_ramp_up},
    target_months=target_months,
)

fte_df = calculator.run()

# CS領域合計を 380FTE に正規化
fte_df = normalize_fte_to_headcount(fte_df, {"CS": 380})

# ============================================================
# 結果表示
# ============================================================

print("=" * 60)
print("【品目×月別 FTE（先頭20行）】")
print("=" * 60)
print(fte_df[["product_id", "month", "target_doctors",
              "visit_frequency", "required_fte"]].head(20).to_string(index=False))

print("\n" + "=" * 60)
print("【年度集計（品目別 平均FTE）】")
print("=" * 60)
summary = calculator.summarize_fy(fte_df)
print(summary.to_string(index=False))

print("\n" + "=" * 60)
print("【領域別合計FTE vs 現行MR数】")
print("=" * 60)
total = calculator.total_fte_by_area(fte_df)
print(total.to_string(index=False))

# ============================================================
# 新品目FTE配分（OVEに必要なFTEを既存品目から削る）
# ============================================================

current_fte = {"GLI": 80.0, "INT": 100.0, "TRI": 90.0}
current_mr_activity = {"GLI": 1200.0, "INT": 1500.0, "TRI": 1350.0}

decay_params_df = pd.DataFrame({
    "product_id": ["GLI", "GLI", "INT", "INT", "TRI", "TRI"],
    "channel":    ["面談", "e-contents", "面談", "e-contents", "面談", "e-contents"],
    "alpha":      [0.7, 0.5, 0.75, 0.55, 0.7, 0.5],
    "beta_m":     [1200.0, 800.0, 1500.0, 1000.0, 1300.0, 900.0],
    "ec_m":       [50.0, 200.0, 60.0, 250.0, 55.0, 220.0],
    "eta_m":      [1.0, 0.8, 1.0, 0.8, 1.0, 0.8],
    "mr_time_weight": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "slope_m":    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
})

allocator = NewProductFTEAllocator(
    decay_params_df=decay_params_df,
    current_fte=current_fte,
    current_mr_activity=current_mr_activity,
    min_fte_ratio=0.5,
)

fte_df2, allocation_df = calculator.run_with_new_product(
    new_product_ids=["OVE"],
    donor_products=["GLI", "INT", "TRI"],
    new_product_fte_allocator=allocator,
)

print("\n" + "=" * 60)
print("【OVE FTE配分（どの品目から何FTE削るか）】")
print("=" * 60)
print(allocation_df.to_string(index=False))
