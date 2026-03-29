"""
FY2029 FTE算出 使用例
=====================
実際のDatabricksデータに差し替えて使用する。
ここではダミーデータで動作確認できる構成にしている。

実行方法（Databricksノートブック）:
  %run ./fy2029_fte_calculator
  # または
  from fy2029_fte_calculator import *
"""

import pandas as pd
import numpy as np
from fy2029_fte_calculator import (
    ProductConfig,
    TargetDoctorCalculator,
    FCScAllocator,
    ActivityFrequencyEstimator,
    MRDigitalRatioEstimator,
    NewProductFTEAllocator,
    FY2029FTECalculator,
    FY2029_MONTHS,
)

# ============================================================
# Step 0: データ読み込み（Databricks環境では spark.read に置き換える）
# ============================================================

# --- 活動データ ---
# 実際: spark.read.table("schema.activity_data").toPandas()
activity_data = pd.DataFrame({
    "activity_ym": ["2028-10", "2028-10", "2028-11", "2028-11",
                    "2028-10", "2028-10", "2028-11", "2028-11"],
    "product_id":  ["TRI", "TRI", "TRI", "TRI",
                    "INT", "INT", "INT", "INT"],
    "doctor_id":   ["D001", "D002", "D001", "D003",
                    "D001", "D004", "D001", "D005"],
    "activity_count": [2, 1, 2, 1, 1, 1, 1, 1],
})

# --- 医師属性 ---
doctor_attr = pd.DataFrame({
    "doctor_id": ["D001", "D002", "D003", "D004", "D005"],
    "age":       [45, 52, 38, 61, 44],
})

# --- 製品情報 ---
product_info = pd.DataFrame({
    "product_id":          ["GLI", "CUV", "HYQ", "INT", "TRI", "OVE"],
    "launch_ym":           ["2018-04", "2020-10", "2022-04",
                            "2019-07", "2021-01", "2029-07"],
    "loe_months":          [180, 144, 132, 168, 120, 240],
    "estimated_patients":  [50000, 30000, 20000, 80000, 60000, 15000],
    "num_indications":     [3, 2, 2, 4, 3, 1],
})

# --- MMMアウトプット: 減衰曲線パラメータ（Meridian出力） ---
# 実際: Meridianの出力テーブルから取得
decay_params_df = pd.DataFrame({
    "product_id": ["GLI","GLI","CUV","CUV","HYQ","HYQ",
                   "INT","INT","TRI","TRI","OVE","OVE"],
    "channel":    ["MR","Digital","MR","Digital","MR","Digital",
                   "MR","Digital","MR","Digital","MR","Digital"],
    "alpha":      [0.7, 0.5, 0.6, 0.4, 0.65, 0.45,
                   0.75, 0.55, 0.7, 0.5, 0.8, 0.6],
    "beta":       [1200, 800, 900, 600, 700, 500,
                   1500, 1000, 1300, 900, 600, 400],
    "ec":         [50, 200, 40, 150, 35, 120,
                   60, 250, 55, 220, 30, 100],
    "slope":      [2.0, 1.5, 1.8, 1.4, 1.9, 1.5,
                   2.1, 1.6, 2.0, 1.5, 1.7, 1.3],
})

# --- MMMアウトプット: 施設別最適実施数テーブル（オプション）---
# 使用する場合はここに読み込む
# mmm_optimal_df = spark.read.table("schema.mmm_optimal_activities").toPandas()
mmm_optimal_df = None

# --- 活動セット品目一覧（FC/SC関係）---
# fc_product を訪問する際、sc_product をセカンドコールとして実施
activity_set_df = pd.DataFrame({
    "fc_product": ["TRI", "INT", "GLI", "GLI"],
    "sc_product": ["INT", "TRI", "CUV", "HYQ"],
})

# --- ターゲット医師リスト（品目別）---
# 実際: 活動データや担当医師マスタから取得
target_doctor_lists = {
    "GLI": pd.Series(["D001","D002","D003","D006","D007"]),
    "CUV": pd.Series(["D001","D002","D008","D009"]),
    "HYQ": pd.Series(["D001","D003","D010"]),
    "INT": pd.Series(["D001","D004","D005","D011","D012"]),
    "TRI": pd.Series(["D001","D002","D004","D013","D014"]),
    "OVE": pd.Series(["D001","D004","D015","D016"]),  # 新品目: 推定リスト
}

# --- 現在の活動量（MR月次コール数 / Digital月次視聴数）---
current_activities = {
    "GLI": {"MR": 5000, "Digital": 8000},
    "CUV": {"MR": 3000, "Digital": 5000},
    "HYQ": {"MR": 2000, "Digital": 3000},
    "INT": {"MR": 6000, "Digital": 10000},
    "TRI": {"MR": 5500, "Digital": 9000},
    "OVE": {"MR":  500, "Digital":  500},  # 新品目: 発売前なので少量
}

# --- 現在のFTE（品目別推定）---
# 実際: FY2028の算出結果から取得
current_fte = {
    "GLI": 30.0,
    "CUV": 20.0,
    "HYQ": 15.0,
    "INT": 40.0,
    "TRI": 35.0,
}

# ============================================================
# Step 1: 品目設定
# ============================================================

product_configs = [
    ProductConfig("GLI", "CS", False, "2018-04", 180, 50000, 3),
    ProductConfig("CUV", "CS", False, "2020-10", 144, 30000, 2),
    ProductConfig("HYQ", "CS", False, "2022-04", 132, 20000, 2),
    ProductConfig("INT", "CS", False, "2019-07", 168, 80000, 4),
    ProductConfig("TRI", "CS", False, "2021-01", 120, 60000, 3),
    ProductConfig("OVE", "CS", True,  "2029-07", 240, 15000, 1),  # 新発売
]

# ============================================================
# Step 2: 各モジュールを初期化
# ============================================================

target_doctor_calc = TargetDoctorCalculator(
    activity_data=activity_data,
    doctor_attr=doctor_attr,
    product_info=product_info,
)

fc_sc_allocator = FCScAllocator(
    activity_set_df=activity_set_df,
    target_doctor_lists=target_doctor_lists,
)

freq_estimator = ActivityFrequencyEstimator(
    activity_data=activity_data,
    product_info=product_info,
    mmm_optimal_activities=mmm_optimal_df,
)

mr_digital_estimator = MRDigitalRatioEstimator(
    decay_params_df=decay_params_df,
)

# ============================================================
# Step 3: FY2029 FTE算出
# ============================================================

# OVEのランチカーブ（発売後0〜11ヶ月の浸透率）
# FY2029-07発売なので、FY2029内は発売後0〜8ヶ月相当
ove_ramp_up = [
    0.05, 0.12, 0.22, 0.35,   # 発売後1〜4ヶ月
    0.48, 0.58, 0.65, 0.70,   # 5〜8ヶ月
    0.74, 0.77, 0.80, 0.82,   # 9〜12ヶ月
]

calculator = FY2029FTECalculator(
    product_configs=product_configs,
    target_doctor_calc=target_doctor_calc,
    fc_sc_allocator=fc_sc_allocator,
    freq_estimator=freq_estimator,
    mr_digital_estimator=mr_digital_estimator,
    product_info=product_info,
    current_activities=current_activities,
    frequency_mode="lifecycle_adjusted",  # 実績 + ライフサイクル補正
    new_product_ramp_up={"OVE": ove_ramp_up},
    reference_products={"OVE": "TRI"},    # OVEの参照品目 = TRI（類似品）
)

# ============================================================
# Step 4: 新品目FTE配分（OVEに必要なFTEをどこから削るか）
# ============================================================

new_product_fte_allocator = NewProductFTEAllocator(
    decay_params_df=decay_params_df,
    current_fte=current_fte,
    current_mr_activity={pid: current_activities[pid]["MR"] for pid in current_fte},
    min_fte_ratio=0.5,  # 各品目は現在FTEの50%以上を維持
)

# 新品目含めて算出 + ドナー品目決定
fte_df, allocation_df = calculator.run_with_new_product(
    new_product_ids=["OVE"],
    donor_products=["GLI", "CUV", "HYQ", "INT", "TRI"],
    new_product_fte_allocator=new_product_fte_allocator,
)

# ============================================================
# Step 5: 結果確認
# ============================================================

print("=" * 60)
print("【FY2029 品目×月別 FTE（全品目）】")
print("=" * 60)
print(fte_df[[
    "product_id", "month", "target_doctors",
    "fc_doctors", "sc_doctors",
    "required_fte", "adjusted_fte",
    "mr_fte", "digital_fte", "mr_ratio"
]].to_string(index=False))

print("\n" + "=" * 60)
print("【FY2029 年度集計（品目別 平均FTE / MR比率）】")
print("=" * 60)
summary = calculator.summarize_fy(fte_df)
print(summary.to_string(index=False))

print("\n" + "=" * 60)
print("【OVE新発売に伴うFTE配分（どの品目から削るか）】")
print("=" * 60)
print(allocation_df.to_string(index=False))

print("\n" + "=" * 60)
print("【領域別合計FTE vs 現行MR数】")
print("=" * 60)
total_fte = calculator.total_fte_by_area(fte_df)
print(total_fte.to_string(index=False))

# ============================================================
# Step 6: （参考）MMMベース頻度との比較（裏計算）
# ============================================================
# FY2026-2028の数値は変更しない。あくまで参考として保持。

if mmm_optimal_df is not None:
    print("\n" + "=" * 60)
    print("【参考: MMM最適頻度ベースのFTE（実績数値には反映しない）】")
    print("=" * 60)

    calculator_mmm = FY2029FTECalculator(
        product_configs=product_configs,
        target_doctor_calc=target_doctor_calc,
        fc_sc_allocator=fc_sc_allocator,
        freq_estimator=freq_estimator,
        mr_digital_estimator=mr_digital_estimator,
        product_info=product_info,
        current_activities=current_activities,
        frequency_mode="mmm_optimal",   # MMMベースに切り替え
        new_product_ramp_up={"OVE": ove_ramp_up},
        reference_products={"OVE": "TRI"},
    )
    fte_df_mmm = calculator_mmm.run()
    summary_mmm = calculator_mmm.summarize_fy(fte_df_mmm)
    print(summary_mmm.to_string(index=False))
